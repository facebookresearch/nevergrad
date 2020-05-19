# Installation and configuration

## Installation of Anaconda and nevergrad

One simple way to work with/on nevergrad is to use anaconda.

First, we have to install Anaconda for Windows.

Please download from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

the version accordingly to your Windows, or for a windows 64 bits you can directly click [here](https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe)

Click on the downloaded file to install it. You can let all proposed options by default.
For the installation all you have to do is to click on "*next*" and "*install*", and at the end "*finish*".

Once this is installed, we have to launch it. Click on the windows key on your keyboard 

![windows key](resources/windows_screenshots/CtrlWindowsAlt.jpg)

and start to dial on "*anaconda*" until "*anaconda navigator*" appears. Click on it.

![Anaconda navigator](resources/windows_screenshots/anacondanavigator.PNG)

Once the navigator is laucnhed, you should get this:

![Anaconda navigator](resources/windows_screenshots/navigator.PNG)

The navigator is the main menu. From here, you can install, launch different softs, create or launch different environment and so on.

We will create a fresh new environment for nevergrad. In the left menu, click on "*environments*" and then on create at the bottom.

You can choose the name you want, and then click on "*create*". 

![Creating environment](resources/windows_screenshots/create.PNG)

Now go back to the main page of Anaconda navigator (by clicking on *Home* in the left menu). 
You can see that you are now in the new new environment. No soft are installed, we will install "*Powershell Prompt*", by clicking on the install button under it.


Once installed, this buttun becomes a "launch" buttun, then just launch it.
You have now a terminal launched. 

![Prompt](resources/windows_screenshots/prompt.PNG)


Dial on 
```
mkdir repos 
cd repos
```

This creates a repository names "*repos*" and goes into it.
Then we will clone nevergrad:
```
git clone https://github.com/facebookresearch/nevergrad.git
```
If you have a fork, you can clone your fork.
For instance, I have one, then I do
``` 
git clone https://github.com/fteytaud/nevergrad.git 
```

Then, to install all the needed packages, follow these commands:

```
cd nevergrad/requirements/
pip install -r main.txt
pip install -r bench.txt
pip install -r dev.txt
conda install pytorch
cd ..
```

Installation in now over

## Adding spyder

**Spyder** is a python ide well integrated in the anaconda environment. If you no other ide, I suggest to install it in order to modify/add/read code from nevegrad.

To install it: In the anaconda environment select you nevergrad environment (in the "application on" menu). Then you can install by clicking on the install button under the spyder case.
You can see other softs you are used to (jupyter, rstudio for instance). The anaconda navigator is the place you can install and launch all your softs.

![Spyder](resources/windows_screenshots/spyder.PNG)

## To run an experiment

Click on the windows key on your keyboard and launch anaconda navigator, and select you nevergrad environment (in the "application on" menu).

Next, launch a powershell prompt and go in the nevergrad repository by dialing
```
cd repos/nevergrad
```

and launch 

```
python -m nevergrad.benchmark parallel to run the parallel experiment
```

## To make a pull request

Knowing how to add a pull request is very useful.
For instance, I want to pull request this documentation.

Click on the windows key on your keyboard and launch anaconda navigator, and select you nevergrad environment (in the "application on" menu).

Next, launch a powershell prompt and go in the nevergrad repository by typing 
```
cd repos/nevergrad
```

I want to create a new branch *windowsDoc*:
```
git checkout -b windowsDoc
```

![Creating a new branch](resources/windows_screenshots/newBranch.PNG)

Then, I add the new files:
```
git add .\docs\windows.md
git add .\docs\resources\windows_screenshots\*
```

and I commit the modification
```
git commit -am "windows documentation"
```

![Commit](resources/windows_screenshots/commit.PNG)

and I push my commit
```
git push --set-upstream origin windowsDoc
```
