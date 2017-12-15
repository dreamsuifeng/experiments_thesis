------------------------------------------------------------------------------------------
	           Readme for the LACU-SVM Package
	 		       version Apr 17, 2014
------------------------------------------------------------------------------------------

The package includes the MATLAB code of LACU-SVM method for learning with augmented class by exploiting unlabeled Data[1]. It employs the SMO solver implemented in svqp2[2].

[1] Qing Da, Yang Yu, and Zhi-Hua Zhou. Learning with Augmented Class by Exploiting Unlabeled Data. In: Proceedings of the 28th AAAI Conference  on Artificial Intelligence (AAAI'14), Qu¨¦bec city, Canada, 2014.
[2] Bottou, L., and Lin, C.-J. 2007. Support vector machine solvers. Large Scale Kernel Machines, 301-320.

For LACU-SVM, you can call the "lacusvm_train" function to train and call the "lacusvm_predict" function to predict. You will find a runnable script of using this code in the "demo.m" file. The mex files of the current version are for linux-64bit machines.

ATTN: 
- This package is free for academic usage. You can run it at your own risk. For other
  purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).

- This package was developed by Mr. Qing Da (daq@lamda.nju.edu.cn). For any
  problem concerning the code, please feel free to contact Mr. Da.

------------------------------------------------------------------------------------------