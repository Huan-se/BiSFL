import tensorflow as tf
import numpy as np


def repack_model(model, layer_idx, kernel_idx):
    canary_layer = model.layers[layer_idx]
    print("---->", canary_layer.name, canary_layer.output.shape)
    canary_layer = canary_layer.output[:, :, :, kernel_idx]
    print("-------->", canary_layer.shape)
    _model = tf.keras.Model([model.input], [model.output, canary_layer])
    return _model


def get_preCanaryTrainable_variables_Conv2D(model, layer_idx):
    # get trainable variables before canary kernel
    pre_canary_layer_trainable_variables = []
    
    for l in model.layers[:layer_idx]:
        var = l.trainable_variables
        print(l.name)
        for v in var:
            print('\t', v.name)
        print()
        pre_canary_layer_trainable_variables += var
        
    return pre_canary_layer_trainable_variables
    
def enumerate_layes(model):
    for i, l in enumerate(model.layers):
        print(i, '\t', l.name, l.output.shape)
        
def get_canary_gradient(G, g_canary_shift, kernel_idx):
    g = G[g_canary_shift]
    if len(g.shape) == 1:
        return g[kernel_idx]
    elif len(g.shape) == 4:
        return g[:, :, :, kernel_idx]
              
def get_gradient(x, y, model, loss_function, variables):
    with tf.GradientTape() as tape:
        y_, att = model(x, training=True)
        loss = loss_function(y, y_)
    g = tape.gradient(loss, variables)
    return [gg.numpy() for gg in g], att


# def local_training(
#     model,
#     training_set,
#     num_iter,
#     learning_rate,
#     loss_function,
#     test_canary_fn
# ):
#     opt = tf.keras.optimizers.SGD(learning_rate)
#     canary_scores = []
#     for i, (x, y) in enumerate(training_set):
#         with tf.GradientTape() as tape:
#             y_, att = model(x, training=True)
#             loss = loss_function(y, y_)
#         g = tape.gradient(loss, model.trainable_variables)
#         opt.apply_gradients(zip(g, model.trainable_variables))
        
#         score, _ = test_canary_fn(model, training_set)
#         canary_scores.append(score)
#         print(f' FedAVG round: {i+1}\n\t{score}')
        
#         if i == num_iter-1:
#             break
#     return canary_scores
def local_training(
    model,
    training_set,
    num_iter,
    learning_rate,
    loss_function,
    test_canary_fn,

):
    mask_scale=1e6
    opt = tf.keras.optimizers.SGD(learning_rate)
    canary_scores = []
    
    for i, (x, y) in enumerate(training_set):
        with tf.GradientTape() as tape:
            y_, att = model(x, training=True)
            loss = loss_function(y, y_)
        
        # 1. 计算原始梯度 g (用于本地更新)
        g = tape.gradient(loss, model.trainable_variables)
        
        # --- 最小化修改：直接加密模型参数 ---
        
        # A. 备份当前真实的模型参数（防止训练崩掉）
        original_weights = [tf.identity(v) for v in model.trainable_variables]
        
        # B. 给模型参数加上巨大的随机掩码
        for var in model.trainable_variables:
            mask = tf.random.uniform(var.shape, -mask_scale, mask_scale, dtype=var.dtype)
            var.assign_add(mask) # 直接修改模型内部参数
            
        # C. 此时模型已是“乱码”，传进去评分，拿到的必然是无效分
        score, _ = test_canary_fn(model, training_set)
        canary_scores.append(score)
        
        # D. 【关键】立刻还原模型参数，确保接下来的本地训练正常
        for var, orig in zip(model.trainable_variables, original_weights):
            var.assign(orig)
            
        # --- 加密逻辑结束 ---

        # 2. 应用原始梯度更新模型（不受上面掩码的影响）
        opt.apply_gradients(zip(g, model.trainable_variables))
        
        print(f' Round: {i+1} | Score: {score}')
        if i == num_iter-1: break
            
    return canary_scores