Please follow the below instructions to Run the solution

NOTE: We have not used any sample PDF or DOC file or fixed data to execute the application but we have dynamically created eml files with attachments of PDF and DOC on run time along with dynamic email body content through code.

1. Install Python Latest Version For Windows or Linux (For Us We have used Winodws latest version 3.13.2)

2. After that please execute the below commends to load the required modules

    pip install reportlab PyPDF2 python-docx transformers torch
    pip install scikit-learn
    pip install pdfminer.six python-docx
    pip install transformers torch pytesseract pillow pandas numpy
    pip install logging shutil datasets sklearn json random

3. We have Implemented 2 Ways of Classification --> PreTrained Model and Fine Tuned Model on local data set

4. For PreTrained Model We have used GPT2 and For Fine Tune Model we have use hugging face BERT.

5. GPT2_Pretrained Model Execution Flow

    i. Open GPT2_Pretrained folder and make sure no file is throwing error for module missing after executing above steps.

    ii. For Single Request Content based classifcation follow the below instructions

         a. Based on number of files want to generate and classify open Config.json file and go to processing section and update num_emails with the required value.

         b. Now execute generate_emails.py file following python command (For VS Code python generate_emails.py) provided folder path is correctly selected.

         c. This generate number of eml files with either PDF or DOC attachment with dynamic body sturcture under one dynamically generated folder email_files along with one log file will be generated where for each file generation activities will be logged.

         d. Now to classify execute classify_emails_with_confidence_score.py file following python command (For VS Code python classify_emails_with_confidence_score.py) provided folder path is correctly selected. (Before execution point the path to email_files folder).

         e. once the file execution will be completed we will see we will be seeing one dynamically generated folder called Classification and under this we will be seeing all request type folders based on classification from the above code.

         f. Classfication folders will content dynamically generated email folders and eml files and alomng with classification json which will content the primary intent value using which classification has happened and extracted fields and Request Type and Sub Request Type value.

         g. This will help user to uniquely check the emails and its contents.

    iii. For Multi Request Content based classifcation follow the below instructions

         a. Based on number of files per type you want to generate and classify open Config_advanced.json file and go to processing section and update num_emails_per_type with the required value.

         b. Now execute generate_emails_advanced.py file following python command (For VS Code python generate_emails_advanced.py) provided folder path is correctly selected.

         c. This generate number of eml files with either PDF or DOC or conatins both DOC and PDF attachment with dynamic body sturcture under one dynamically generated folder email_files and inside this type folder will be generated basded on configured on the Config_Advaced.json file along with one log file will be generated where for each file generation activities will be logged.

         d. Now to classify execute classify_emails_advanced_with_confidence_score.py file following python command (For VS Code python classify_emails_advanced_with_confidence_score.py) provided folder path is correctly selected. (Before execution point the path to email_files folder).

         e. once the file execution will be completed we will see we will be seeing one dynamically generated folder called Classification and under this we will be seeing all type folder are getting generated and under those type folders request type folders based on classification from the above code gets generated.

         f. Classfication folders will content dynamically generated type folders under which dynamically generated email folders under which corresponding eml file and along with classification json which will contain the primary intent value using which classification has happened and extracted fields and all the Request Types and Sub Request Types value.

         g. This will help user to uniquely check the emails and its contents.

6. BERT_Pretrained_Fine_Tune Model Execution Flow

    i. Open BERT_Pretrained_Fine_Tune folder and make sure no file is throwing error for module missing after executing above steps.

    ii. First you need to train the model using fine_tune_bert.py file and using request_type_model, sub_request_type_model folder and trainer details folder data and files.

    iii. For Single Request Content based classifcation follow the below instructions

         a. Based on number of files want to generate and classify open Config.json file and go to processing section and update num_emails with the required value.

         b. Now execute generate_emails.py file following python command (For VS Code python generate_emails.py) provided folder path is correctly selected.

         c. This generate number of eml files with either PDF or DOC attachment with dynamic body sturcture under one dynamically generated folder email_files along with one log file will be generated where for each file generation activities will be logged.

         d. Now to classify execute classify_email_bert_confidencescore.py file following python command (For VS Code python classify_email_bert_confidencescore.py) provided folder path is correctly selected. (Before execution point the path to email_files folder).

         e. once the file execution will be completed we will see we will be seeing one dynamically generated folder called Classification and under this we will be seeing all request type folders based on classification from the above code.

         f. Classfication folders will content dynamically generated email folders and eml files and alomng with classification json which will content the primary intent value using which classification has happened and extracted fields and Request Type and Sub Request Type value.

         g. This will help user to uniquely check the emails and its contents.

    iv. For Multi Request Content based classifcation follow the below instructions

         a. Based on number of files per type you want to generate and classify open Config_advanced.json file and go to processing section and update num_emails_per_type with the required value.

         b. Now execute generate_emails_advanced.py file following python command (For VS Code python generate_emails_advanced.py) provided folder path is correctly selected.

         c. This generate number of eml files with either PDF or DOC or conatins both DOC and PDF attachment with dynamic body sturcture under one dynamically generated folder email_files and inside this type folder will be generated basded on configured on the Config_Advaced.json file along with one log file will be generated where for each file generation activities will be logged.

         d. Now to classify execute classify_email_bert_advanced_confidencescore.py file following python command (For VS Code python classify_email_bert_advanced_confidencescore.py) provided folder path is correctly selected. (Before execution point the path to email_files folder).

         e. once the file execution will be completed we will see we will be seeing one dynamically generated folder called Classification and under this we will be seeing all type folder are getting generated and under those type folders request type folders based on classification from the above code gets generated.

         f. Classfication folders will content dynamically generated type folders under which dynamically generated email folders under which corresponding eml file and along with classification json which will contain the primary intent value using which classification has happened and extracted fields and all the Request Types and Sub Request Types value.

         g. This will help user to uniquely check the emails and its contents.
