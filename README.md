Data Science
## Spring 2025 Edition

This repository contains the materials for the Spring 2025 edition of the Erdős Institute's Data Science Boot Camp programming

[https://www.erdosinstitute.org/programs/spring-2025/data-science-boot-camp](https://www.erdosinstitute.org/programs/spring-2025/data-science-boot-camp) 

## conda environment

You should set up an erdos_spring_2025 conda environment and run all of the notebooks with this environment. Below are minimial instructions.  You can find more detailed instructions here: [https://docs.google.com/document/d/1UsmDLnQwGjTB7eLKoccQZvLq2uWJexFcNov_CzkN2Ng/edit?tab=t.0](https://docs.google.com/document/d/1UsmDLnQwGjTB7eLKoccQZvLq2uWJexFcNov_CzkN2Ng/edit?tab=t.0)

Check to make sure you have conda by running the following in your command line interface:

    conda --version

If you don't have conda, google how to install it!

Once you have conda run:

    conda update conda

Press [y] to all of the prompts.  When you are done check the version again.  It should be version 23 or above.

Then run 

    conda env create --file=environment.yml

Press [y] to all of the prompts.  You will be downloading a lot of packages.

Once this is done:

    conda activate erdos_spring_2025

To check everything is there:

    conda list

Should show all of the packages!

If you ever need to add more packages to this environment please refer to the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages).

Finally open `computer_setup_day/find_secret_code` and run the only cell there.  It will give you a numerical output.  Put that output in the textbox at [https://www.erdosinstitute.org/ds-boot-camp-prep](https://www.erdosinstitute.org/ds-boot-camp-prep).  If you set up the conda env correctly it will tell you that you have it set up correctly!  Please do this because us figure out who still needs assistance setting up python/jupyter/conda.

If you are using Visual Studio Code just select erdos_spring_2025 as the kernel when running the notebooks.

## Data

The data folder contains various data files used throughout the repository's notebooks.

## Weeks

These folders contain all of the materials which we will cover in a given week of the bootcamp.  They are structured as follows:

    .
    └── week-n-descriptive-name/
        ├── lecture-n
        ├── problem-session-n
        ├── math-n
        ├── optional-extra-practice-n
        └── optional-additional-content-n


Every week will have at least the lecture and problem-session folders.  Some weeks also have one or more of the additional folders.

### lecture-n

These folders contain the lecture jupyter notebooks for the boot camp. Each lecture will have a blank version of the notebook as well as a "complete" version of the notebook. The complete notebooks match the notebooks seen in the pre-recorded lecture videos on the boot camp's website. The blank versions are for you to complete on your own. Feel free to practice or take notes with these versions of the notebooks. 

### problem-session-n

These folders contain the jupyter notebooks used in our problem solving sessions.  Note that problem solving sessions are **synchronous**. You can also find "prep" notebooks in this folder. Prep notebooks are for those participants wishing to have additional practice prior to the correspong problem solving session. They will remind you of some python fundamentals which will be needed to complete the problem session.  Prep notebooks are optional.

### math-n

These folders contain supplemental expositions of the mathematics underlying some of the content from the lecture.  They are also optional.  We will discuss these notebooks during "math hour".

### optional-extra-practice-n

These folders contain jupyter notebooks that hold problems you can work on for practice. Each notebook has an accompanying solution notebook. <i>Note</i>, these notebooks are not mandatory, they just exist as a way for you to test yourself on the material covered in lecture.  Think of them as an extra **asynchronous** problem session you can use just to build up your skills.

### optional-additional-content-n

These folders contain notebooks which are similar to the lecture notebooks, but cover content we will not be able to fit into the live bootcamp.


-------------------------
Copyright Info

This repository was written for the 2024 Erdős Institute Data Science Boot Camp by Matthew Osborne, Ph. D., in 2023 and modified by Steven Gubkin in 2024.

Any potential redistributors must seek and receive permission from Matthew Tyler Osborne and Steven Gubkin prior to redistribution. Redistribution of the material contained in this repository is conditional on acknowledgement of Matthew Tyler Osborne original authorship and sponsorship of the Erdős Institute. (see License.md)
