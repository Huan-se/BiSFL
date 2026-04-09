# BiSFL: Bidirectionally Secure Federated Learning Framework via Trusted Execution Environments
This is the repo of the paper BiSFL, which is devided into two part:
### Complete_Implementation
This contains an overall implementation of BiSFL but just with a random generator of '''std::mt19937'''.

### TEST
This contains all of the tests used in BiSFL's paper.
#### Effect_Evaluation_Train
A plaintext version used to train for the Acc and ASR.
#### Effect_Evaluation_Train_IG_and_Canary
An adapted version used to verify the effect of defence of IG and Canary attack.
#### Overall_Overhead
An overall implementation with detailed test of BiSFL with a fast AES-CTR-128 random generator. Meanwhile, I reproduce the experiments of the baseline schemes including ShieldFL, Lancelot(fix their origin repo) and EPPRFL.
#### Performance_Evaluation_Dection
An adapted version used to evaluate the cost of detection method, but I discard this cause some unrepairable bugs.
#### Performance_Evaluation_SA
An adapted version used to evaluate the cost of SA phase without the former projection and train phase.
