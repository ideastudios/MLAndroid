/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package tech.oom.mlandroid;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity {

    public static INDArray output;
    MainActivity.DrawingView drawingView;
    String absolutePath;

    //helper class to return the largest value in the output array
    public static double arrayMaximum(double[] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for (double cur : arr)
            max = Math.max(max, cur);
        return max;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        RelativeLayout parent = findViewById(R.id.layout2);
        drawingView = new MainActivity.DrawingView(this);
        parent.addView(drawingView);

        layer();
//        iris();


    }

    private void layer() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                DenseLayer inputLayer = new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(3)
                        .name("Input")
                        .build();
                DenseLayer hiddenLayer = new DenseLayer.Builder()
                        .nIn(3)
                        .nOut(2)
                        .name("Hidden")
                        .build();
                OutputLayer outputLayer = new OutputLayer.Builder()
                        .nIn(2)
                        .nOut(2)
                        .name("Output")
                        .build();



                NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
                nncBuilder.updater(Updater.ADAM);
                NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
                listBuilder.layer(0, inputLayer);
                listBuilder.layer(1, hiddenLayer);
                listBuilder.layer(2, outputLayer);

                MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
                myNetwork.init();

                INDArray trainingInputs = Nd4j.zeros(4, inputLayer.getNIn());
                INDArray trainingOutputs = Nd4j.zeros(4, outputLayer.getNOut());

                // If 0,0 show 0
                trainingInputs.putScalar(new int[]{0, 0}, 0);
                trainingInputs.putScalar(new int[]{0, 1}, 0);
                trainingOutputs.putScalar(new int[]{0, 0}, 0);
// If 0,1 show 1
                trainingInputs.putScalar(new int[]{1, 0}, 0);
                trainingInputs.putScalar(new int[]{1, 1}, 1);
                trainingOutputs.putScalar(new int[]{1, 0}, 1);
// If 1,0 show 1
                trainingInputs.putScalar(new int[]{2, 0}, 1);
                trainingInputs.putScalar(new int[]{2, 1}, 0);
                trainingOutputs.putScalar(new int[]{2, 0}, 1);
// If 1,1 show 0
                trainingInputs.putScalar(new int[]{3, 0}, 1);
                trainingInputs.putScalar(new int[]{3, 1}, 1);
                trainingOutputs.putScalar(new int[]{3, 0}, 0);

                DataSet myData = new DataSet(trainingInputs, trainingOutputs);
                for(int l=0; l<=1000; l++) {
                    myNetwork.fit(myData);
                }

            }
        }).start();
    }

    private void iris() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                //Get the doubles from params, which is an array so they will be 0,1,2,3


                double pld = 1;

                double pwd = 1;

                double sld = 1;

                double swd = 1;


                //Write them in the log

                Log.d("myNetwork Output ", "do in background string pl = " + pld);

                Log.d("myNetwork Output ", "do in background string pw = " + pwd);

                Log.d("myNetwork Output ", "do in background string sl = " + sld);

                Log.d("myNetwork Output ", "do in background string sw = " + swd);


                //Create input

                INDArray actualInput = Nd4j.zeros(1, 4);

                actualInput.putScalar(new int[]{0, 0}, pld);

                actualInput.putScalar(new int[]{0, 1}, pwd);

                actualInput.putScalar(new int[]{0, 2}, sld);

                actualInput.putScalar(new int[]{0, 3}, swd);

                INDArray ones = Nd4j.ones(100, 10);
                System.out.println("ones is " + ones);


                //Convert the iris data into 150x4 matrix

                int row = 150;

                int col = 4;


                double[][] irisMatrix = new double[row][col];

                int i = 0;

                for (int r = 0; r < row; r++) {

                    for (int c = 0; c < col; c++) {

                        irisMatrix[r][c] = IrisDataSet.irisData[i++];

                    }

                }


                //Check the array by printing it in the log

                System.out.println(Arrays.deepToString(irisMatrix).replace("], ", "]\n"));


                //Now do the same for the label data

                int rowLabel = 150;

                int colLabel = 3;


                double[][] twodimLabel = new double[rowLabel][colLabel];

                int ii = 0;

                for (int r = 0; r < rowLabel; r++) {

                    for (int c = 0; c < colLabel; c++) {

                        twodimLabel[r][c] = IrisDataSet.labelData[ii++];

                    }

                }


                System.out.println(Arrays.deepToString(twodimLabel).replace("], ", "]\n"));


                //Convert the data matrices into training INDArrays

                INDArray trainingIn = Nd4j.create(irisMatrix);

                INDArray trainingOut = Nd4j.create(twodimLabel);


                //build the layers of the network

                DenseLayer inputLayer = new DenseLayer.Builder()

                        .nIn(4)

                        .nOut(3)

                        .name("Input")

                        .build();


                DenseLayer hiddenLayer = new DenseLayer.Builder()

                        .nIn(3)

                        .nOut(3)

                        .name("Hidden")

                        .build();


                OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)

                        .nIn(3)

                        .nOut(3)

                        .name("Output")

                        .activation(Activation.SOFTMAX)

                        .build();


                NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();

                long seed = 6;

                nncBuilder.seed(seed);

                nncBuilder.activation(Activation.TANH);

                nncBuilder.weightInit(WeightInit.XAVIER);


                NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();

                listBuilder.layer(0, inputLayer);

                listBuilder.layer(1, hiddenLayer);

                listBuilder.layer(2, outputLayer);


                MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());

                myNetwork.init();


                //Create a data set from the INDArrays and train the network

                DataSet myData = new DataSet(trainingIn, trainingOut);

                for (int l = 0; l <= 1000; l++) {

                    myNetwork.fit(myData);

                }


                //Evaluate the input data against the model

                INDArray actualOutput = myNetwork.output(actualInput);

                Log.d("myNetwork Output ", actualOutput.toString());


                //Retrieve the three probabilities

                double first = actualOutput.getDouble(0, 0);

                double second = actualOutput.getDouble(0, 1);

                double third = actualOutput.getDouble(0, 2);

                System.out.println("first=" + first + "second=" + second + "third=" + third);
            }
        }).start();
    }

    public String saveDrawing() {
        drawingView.setDrawingCacheEnabled(true);
        Bitmap b = drawingView.getDrawingCache();

        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        // set the path to storage
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        // Create imageDir and store the file there. Each new drawing will overwrite the previous
        File mypath = new File(directory, "drawn_image.jpg");

        //use a fileOutputStream to write the file to the location in a try / catch block
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            b.compress(Bitmap.CompressFormat.JPEG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return directory.getAbsolutePath();
    }

    private void loadImageFromStorage(String path) {

        //use a fileInputStream to read the file in a try / catch block
        try {
            File f = new File(path, "drawn_image.jpg");
            Bitmap b = BitmapFactory.decodeStream(new FileInputStream(f));
            ImageView img = (ImageView) findViewById(R.id.outputView);
            img.setImageBitmap(b);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    public void onProgressBar() {
        TextView bar = findViewById(R.id.processing);
        bar.setVisibility(View.VISIBLE);
    }

    public void offProgressBar() {
        TextView bar = findViewById(R.id.processing);
        bar.setVisibility(View.INVISIBLE);
    }

    // helper class to find the index (and therefore numerical value) of the largest confidence score
    public int getIndexOfLargestValue(double[] array) {
        if (array == null || array.length == 0) return -1;
        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest;
    }

    private class AsyncTaskRunner extends AsyncTask<String, Integer, INDArray> {

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected INDArray doInBackground(String... params) {
            System.out.println("hahahahah begin");
//            OutputLayer outputLayer = new OutputLayer.Builder()
//                    .nIn(2)
//                    .nOut(1)
//                    .name("Output")
//                    .build();
//            System.out.println("hahahahah end");
            // Main background thread, this will load the model and test the input image
            int height = 28;
            int width = 28;
            int channels = 1;

            //load the model from the raw folder with a try / catch block
            try {
                // Load the pretrained network.
                InputStream inputStream = getResources().openRawResource(R.raw.trained_mnist_model);
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(inputStream);

                //load the image file to test
                File f = new File(absolutePath, "drawn_image.jpg");

                //Use the nativeImageLoader to convert to numerical matrix
                NativeImageLoader loader = new NativeImageLoader(height, width, channels);

                //put image into INDArray
                INDArray image = loader.asMatrix(f);

                //values need to be scaled
                DataNormalization scalar = new ImagePreProcessingScaler(0, 1);

                //then call that scalar on the image dataset
                scalar.transform(image);

                //pass through neural net and store it in output array
                output = model.output(image);

            } catch (IOException e) {
                e.printStackTrace();
            }
            return output;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }


        @Override
        protected void onPostExecute(INDArray result) {
            super.onPostExecute(result);

            //used to control the number of decimals places for the output probability
            DecimalFormat df2 = new DecimalFormat(".##");
            System.out.println("result is " + result);
            //transfer the neural network output to an array
            double[] results = {result.getDouble(0, 0), result.getDouble(0, 1), result.getDouble(0, 2),
                    result.getDouble(0, 3), result.getDouble(0, 4), result.getDouble(0, 5), result.getDouble(0, 6),
                    result.getDouble(0, 7), result.getDouble(0, 8), result.getDouble(0, 9),};

            //find the UI tvs to display the prediction and confidence values
            TextView out1 = findViewById(R.id.prediction);
            TextView out2 = findViewById(R.id.confidence);

            //display the values using helper functions defined below
            out2.setText(String.valueOf(df2.format(arrayMaximum(results))));
            out1.setText(String.valueOf(getIndexOfLargestValue(results)));

            //helper function to turn off progress test
            offProgressBar();
        }

    }

    //code for the drawing input
    public class DrawingView extends View {

        private static final float TOUCH_TOLERANCE = 4;
        private Path mPath;
        private Paint mBitmapPaint;
        private Paint mPaint;
        private Bitmap mBitmap;
        private Canvas mCanvas;
        private float mX, mY;

        public DrawingView(Context c) {
            super(c);

            mPath = new Path();
            mBitmapPaint = new Paint(Paint.DITHER_FLAG);
            mPaint = new Paint();
            mPaint.setAntiAlias(true);
            mPaint.setStrokeJoin(Paint.Join.ROUND);
            mPaint.setStrokeCap(Paint.Cap.ROUND);
            mPaint.setStrokeWidth(60);
            mPaint.setDither(true);
            mPaint.setColor(Color.WHITE);
            mPaint.setStyle(Paint.Style.STROKE);
        }

        @Override
        protected void onSizeChanged(int W, int H, int oldW, int oldH) {
            super.onSizeChanged(W, H, oldW, oldH);
            mBitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_4444);
            mCanvas = new Canvas(mBitmap);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            canvas.drawBitmap(mBitmap, 0, 0, mBitmapPaint);
            canvas.drawPath(mPath, mPaint);
        }

        private void touch_start(float x, float y) {
            mPath.reset();
            mPath.moveTo(x, y);
            mX = x;
            mY = y;
        }

        private void touch_move(float x, float y) {
            float dx = Math.abs(x - mX);
            float dy = Math.abs(y - mY);
            if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
                mPath.quadTo(mX, mY, (x + mX) / 2, (y + mY) / 2);
                mX = x;
                mY = y;
            }
        }

        private void touch_up() {
            mPath.lineTo(mX, mY);
            mCanvas.drawPath(mPath, mPaint);
            mPath.reset();
        }

        @Override
        public boolean onTouchEvent(MotionEvent event) {
            float x = event.getX();
            float y = event.getY();

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    invalidate();
                    clear();
                    touch_start(x, y);
                    invalidate();
                    break;
                case MotionEvent.ACTION_MOVE:
                    touch_move(x, y);
                    invalidate();
                    break;
                case MotionEvent.ACTION_UP:
                    touch_up();
                    absolutePath = saveDrawing();
                    invalidate();
                    clear();
                    loadImageFromStorage(absolutePath);
                    onProgressBar();
                    //launch the asyncTask now that the image has been saved
                    AsyncTaskRunner runner = new AsyncTaskRunner();
                    runner.execute(absolutePath);
                    break;

            }
            return true;
        }

        public void clear() {
            mBitmap.eraseColor(Color.TRANSPARENT);
            invalidate();
            System.gc();
        }

    }

}





