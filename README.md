# Tensorflow-preprocessing-training-testing

(non-distribution & distribution)


Use Tensorflow to do classification containing data preparation, training, testing.(single computer single GPU &amp; single computer multi-GPU &amp; multi-computer multi-GPU)

---

**All parameters are in `arg_parsing.py`. So before you start this program, you should read it carefully!**

**STEPS:**

1. Run **image_processing.py** to convert the images from rgb to tfrecords.

3. For single computer, one GPU or more, whatever. Just run:

        python main.py
  
4. For distribution, first you should modify **PS_HOSTS** and **WORKER_HOSTS** in **arg_parsing.py**. And then copy all dataset and codes to every server. 

5. All ckpt and event files will be saved in **MODEL_DIR**.
6. For testing, just run:

       python src/main.py --mode=testing

---

**Notes**

1. For visualization, run:

       tensorboard --logdir=models/
    
