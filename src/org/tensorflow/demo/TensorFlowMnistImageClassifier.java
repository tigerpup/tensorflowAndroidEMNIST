package org.tensorflow.demo;

/**
 * Created by philips on 20/12/17.
 */


import android.annotation.SuppressLint;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Trace;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.demo.env.ImageUtils;

public class TensorFlowMnistImageClassifier implements Classifier{

    private static final String TAG = "TensorFlowMnistImageClassifier";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.3f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputSize;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] floatValues;
    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    private TensorFlowMnistImageClassifier() {}

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputSize The input size. A square image of inputSize x inputSize is assumed.
     * @param imageMean The assumed mean of the image values.
     * @param imageStd The assumed std of the image values.
     * @param inputName The label of the image input node.
     * @param outputName The label of the output node.
     * @throws IOException
     */
    @SuppressLint("LongLogTag")
    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String outputName) {
        TensorFlowMnistImageClassifier c = new TensorFlowMnistImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;

        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                c.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!" , e);
        }

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputSize = inputSize;
        c.imageMean = imageMean;
        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[] {outputName};
        c.intValues = new int[inputSize * inputSize];
        c.floatValues = new float[inputSize * inputSize];
        c.outputs = new float[numClasses];

        return c;
    }
    @Override
    public List<Classifier.Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
//         bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//        for (int i = 0; i < intValues.length; ++i) {
//            final int val = intValues[i];
//            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
//            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
//            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
//        }
        // Convert the imput bitmap into a 784 vector for BW images using CM
//        Bitmap nBitmap=getBWFromBitmap(bitmap);
//        Bitmap nBitmap=test(bitmap);
//        nBitmap.getPixels(intValues, 0, nBitmap.getWidth(), 0, 0, nBitmap.getWidth(), nBitmap.getHeight());

//        for (int i = 0; i < intValues.length; ++i) {
//            floatValues[i]=intValues[i];
//        }
        floatValues=test(bitmap);
        //dont need to process further
        //convert the pixels to match the input.
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 1);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Classifier.Recognition> pq =
                new PriorityQueue<Classifier.Recognition>(
                        3,
                        new Comparator<Classifier.Recognition>() {
                            @Override
                            public int compare(Classifier.Recognition lhs, Classifier.Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                pq.add(
                        new Classifier.Recognition(
                                "" + i, i <=47 ? i+"" : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Classifier.Recognition> recognitions = new ArrayList<Classifier.Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public List<Classifier.Recognition> recognizeImages(int bitmap[]) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
//         bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//        for (int i = 0; i < intValues.length; ++i) {
//            final int val = intValues[i];
//            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
//            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
//            floatValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
//        }
        // Convert the imput bitmap into a 784 vector for BW images using CM
//        Bitmap nBitmap=getBWFromBitmap(bitmap);
//        Bitmap nBitmap=test(bitmap);
//        nBitmap.getPixels(intValues, 0, nBitmap.getWidth(), 0, 0, nBitmap.getWidth(), nBitmap.getHeight());

//        for (int i = 0; i < intValues.length; ++i) {
//            floatValues[i]=intValues[i];
//        }
        floatValues=testing(bitmap);
        //dont need to process further
        //convert the pixels to match the input.
        Trace.endSection();

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
//        inferenceInterface.feed(inputName, floatValues, -1, inputSize, inputSize, 1);
        inferenceInterface.feed(inputName, floatValues, 1, 784);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        // Find the best classifications.
        PriorityQueue<Classifier.Recognition> pq =
                new PriorityQueue<Classifier.Recognition>(
                        3,
                        new Comparator<Classifier.Recognition>() {
                            @Override
                            public int compare(Classifier.Recognition lhs, Classifier.Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > THRESHOLD) {
                pq.add(
                        new Classifier.Recognition(
                                "" + i, getLetter(i)!=""? getLetter(i) : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Classifier.Recognition> recognitions = new ArrayList<Classifier.Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

//    public static Bitmap getBWFromBitmap(final Bitmap bitmap){
//        Bitmap bmpMonochrome = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
//        Canvas canvas = new Canvas(bmpMonochrome);
//        ColorMatrix ma = new ColorMatrix();
//        ma.setSaturation(0);
//        Paint paint = new Paint();
//        paint.setColorFilter(new ColorMatrixColorFilter(ma));
//        canvas.drawBitmap(bitmap, 0, 0, paint);
//        return bmpMonochrome;
//    }
    public static float[] testing(int pixels[]){
        float[] sample=new float[pixels.length];
        int black=0;int white=0;
        for(int i=0;i<pixels.length;i++){
//            if( pixels[i] >= 0xff ){
//                pixels[i]=0xff;
////                black++;
//            }
//            else if(pixels[i] < 0){
//                pixels[i] = 0;
////                white++;
//            }
            if(pixels[i]>=0x80){
                black++;
                pixels[i]=0xff;
            }
            else{
//                white++;
                pixels[i]=0;

            }
            sample[i]=(float)pixels[i]/255.0f;
        }

        return sample;

    }

    public static int getAscii(int index){
        // Based on EMNIST DATASET. Refer to emnist-balanced-mapping.txt
        // index 0 to 9 : Ascii 48 to 57
        // index 10 to 35 : Ascii 65 to 90
        // index 36 : Ascii 97
        // index 37 : Ascii 98
        // index 38 to 42: Ascii 100 to 104
        // index 43 : Ascii 110
        // index 44 : Ascii 113
        // index 45 : Ascii 114
        // index 46 : Ascii 116

        if(index>=0 && index <10){
            return 48+index;
        }
        else if(index>=10 && index < 36){
            return 65+(index-10);
        }
        else if(index>=38 && index <43){
            return 100+(index-38);
        }
        else{
            //handling indvidual case
            switch (index){
                case 36:
                    return 97;
                case 37:
                    return 98;
                case 43:
                    return 110;
                case 44:
                    return 113;
                case 45:
                    return 114;
                case 46:
                    return 116;
                default:
                    return -1;
            }
        }
    }

    public static String getLetter(int index){
        int ascii=getAscii(index);
        if(ascii>0){
            return Character.toString((char)ascii);
        }
        return "";

    }

    public static float[] test(Bitmap src){
        int width = src.getWidth();
        int height = src.getHeight();
        // create output bitmap
//        Bitmap bmOut = Bitmap.createBitmap(width, height, src.getConfig());
        float sample[]=new float[width*height];
        // color information
        int A, R, G, B;int k=0;
        int pixel;
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                // get pixel color
                pixel = src.getPixel(x, y);
                A = Color.alpha(pixel);
                R = Color.red(pixel);
                G = Color.green(pixel);
                B = Color.blue(pixel);
                int gray = (int) (0.2989 * R + 0.5870 * G + 0.1140 * B);
                // use 128 as threshold, above -> white, below -> black
                if (gray > 128) {
                        gray = 255;
                }
                else{
                    gray = 0;
                }
                sample[k++]=gray/255.0f;
                // set new pixel color to output bitmap
//                bmOut.setPixel(x, y, Color.argb(A, gray, gray, gray));
            }
        }
        return sample;
    }
}
