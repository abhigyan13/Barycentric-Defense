# Barycentric-Defense
[A V Subramanyam](https://iiitd.ac.in/subramanyam),  [Abhigyan Raj](https://www.linkedin.com/in/abhigyan-raj-023537145/)

This is the official implementation of our IEEE ICIP 2022 paper titled Barycentric Defense where we demonstrate and develop how barycenters computed in Wasserstein space defend against adversarial attacks.

# What is in this repository?
1. An implementation of our paper barycentric defense.
2. Code to compute barycenters of MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100.
3. Compute accuracies of all testsets on the proposed defense against attacks like Linfinity, L2. 

# Abstract
Wasserstein metric based adversarial attacks have attracted a great interest in the recent past. Even
though they exhibit strong attacks, surprisingly, they have not been investigated for defense. 
In this work, we demonstrate that barycenters computed in Wasserstein space can act as a measure of defense
against adversarial attacks. We compute the barycenter using marginals obtained from the
given image and demonstrate its effectiveness in defense even without any adversarial training. We further analyse
the barycenters using GradCam to understand their defensive characteristics. Elaborate experiments on MNIST,
Fashion-MNIST, CIFAR-10 and CIFAR-100 demonstrate a significant increase in the robustness of victim classifiers.
