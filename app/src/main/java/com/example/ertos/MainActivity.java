package com.example.ertos;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements SensorEventListener {

    // UI elements to display sensor data and predictions
    private TextView xText, yText, zText, predictionText;

    // Sensor management objects
    private Sensor accelerometer;
    private SensorManager sensorManager;

    // Kalman Filter instances for each axis (X, Y, Z)
    private KalmanFilter kalmanFilterX;
    private KalmanFilter kalmanFilterY;
    private KalmanFilter kalmanFilterZ;

    // TensorFlow Lite model interpreter
    private Interpreter tflite;

    // Normalization parameters (mean and standard deviation for each axis)
    private float[] meanValues = {0f,  2f,  0f};  // Adjust these based on your training data
    private float[] stdValues = {0f, 2f, 0f};     // Adjust these based on your training data

    // Activity label mapping (must match training labels)
    private String[] activityLabels = {"Sitting", "Walking", "Running"};

    // Handler for periodic predictions
    private Handler predictionHandler = new Handler();

    // Timing and filtered values storage
    private long lastUpdateTime = 0;
    private float filteredX, filteredY, filteredZ;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize Kalman filters with specified parameters:
        // q (process noise), r (measurement noise), initial value, initial covariance
        kalmanFilterX = new KalmanFilter(0.2f, 0.7f, 0.0f, 0.7f);
        kalmanFilterY = new KalmanFilter(0.2f, 0.7f, 0.0f, 0.7f);
        kalmanFilterZ = new KalmanFilter(0.2f, 0.7f, 0.0f, 0.7f);

        initializeViews();
        initializeSensor();
        loadModel();
        startPredictionCycle();
    }

    /**
     * Initialize UI components
     */
    private void initializeViews() {
        xText = findViewById(R.id.xText);          // TextView for X-axis value
        yText = findViewById(R.id.yText);          // TextView for Y-axis value
        zText = findViewById(R.id.zText);          // TextView for Z-axis value
        predictionText = findViewById(R.id.predictionText);  // TextView for activity prediction
        predictionText.setText("Activity: Unknown");  // Default prediction text
    }

    /**
     * Initialize accelerometer sensor
     */
    private void initializeSensor() {
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            if (accelerometer != null) {
                // Register listener with normal sampling rate
                sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            } else {
                Toast.makeText(this, "No accelerometer sensor found", Toast.LENGTH_SHORT).show();
            }
        }
    }

    /**
     * Load TensorFlow Lite model from assets
     */
    private void loadModel() {
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    /**
     * Helper method to load TFLite model file from assets
     */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("activity_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Start periodic prediction cycle (every 2 seconds)
     */
    private void startPredictionCycle() {
        predictionHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                // Predict activity using filtered values
                predictActivity(filteredX, filteredY, filteredZ);
                // Schedule next prediction
                predictionHandler.postDelayed(this, 2000); // 2 seconds interval
            }
        }, 2000); // Initial delay of 2 seconds
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        long currentTime = System.currentTimeMillis();
        // Throttle sensor updates to ~100Hz (10ms interval)
        if (currentTime - lastUpdateTime > 10) {
            lastUpdateTime = currentTime;

            // Raw accelerometer values
            float x = event.values[0];
            float y = event.values[1];
            float z = event.values[2];

            // Apply Kalman filter to each axis
            kalmanFilterX.predict();
            kalmanFilterX.update(x);
            kalmanFilterY.predict();
            kalmanFilterY.update(y);
            kalmanFilterZ.predict();
            kalmanFilterZ.update(z);

            // Store filtered values
            filteredX = (float) kalmanFilterX.getEstimate();
            filteredY = (float) kalmanFilterY.getEstimate();
            filteredZ = (float) kalmanFilterZ.getEstimate();

            // Update UI with filtered values
            xText.setText(String.format(Locale.getDefault(), "X: %.2f", filteredX));
            yText.setText(String.format(Locale.getDefault(), "Y: %.2f", filteredY));
            zText.setText(String.format(Locale.getDefault(), "Z: %.2f", filteredZ));
        }
    }

    /**
     * Predict activity using the TensorFlow Lite model
     * @param x Filtered X-axis value
     * @param y Filtered Y-axis value
     * @param z Filtered Z-axis value
     */
    private void predictActivity(float x, float y, float z) {
        // Normalize input using precomputed mean and std
        float[] input = new float[3];
        input[0] = (x - meanValues[0]) / stdValues[0];
        input[1] = (y - meanValues[1]) / stdValues[1];
        input[2] = (z - meanValues[2]) / stdValues[2];

        // Output array for model predictions
        float[][] output = new float[1][3];

        // Run inference
        tflite.run(input, output);

        // Get predicted class (index with highest probability)
        int predictedClass = argmax(output[0]);
        String activity = activityLabels[predictedClass];

        // Update UI with prediction
        predictionText.setText("Activity: " + activity);
    }

    /**
     * Helper method to find index of maximum value in array
     * @param array Input array
     * @return Index of maximum value
     */
    private int argmax(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not used for accelerometer
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Unregister sensor listener to save battery
        sensorManager.unregisterListener(this);
        // Remove pending prediction callbacks
        predictionHandler.removeCallbacksAndMessages(null);
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Re-register sensor listener when activity resumes
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Clean up TensorFlow Lite interpreter
        if (tflite != null) {
            tflite.close();
        }
    }

    /**
     * Kalman Filter implementation for sensor noise reduction
     */
}