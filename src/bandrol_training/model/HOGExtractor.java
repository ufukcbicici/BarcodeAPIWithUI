package bandrol_training.model;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import javafx.scene.control.Tab;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static bandrol_training.Constants.DEBUGPATH;

public class HOGExtractor
{
    private static final int WINDOW_WIDTH = 40;
    private static final int WINDOW_HEIGHT = 56;
    private static final int PATCH_SIZE = 8;
    private static final int BLOCK_SIZE = 16;
    private static final int BIN_COUNT = 9;

    private static Mat[] getGradients(Mat img)
    {
        Mat convertedToReal = new Mat();
        // Imgcodecs.imwrite(DEBUGPATH + "bb.png", img);
        img.convertTo(convertedToReal, CvType.CV_64F, 1.0);

        // Step 1: Split into RGB channels.
        List<Mat> channels = new ArrayList<>();
        Core.split(convertedToReal, channels);

        // Step 2: Calculate gradients at each pixel for each channel
        ArrayList<Mat>[] grads = new ArrayList[2];
        grads[0] = new ArrayList<Mat>();
        grads[1] = new ArrayList<Mat>();

        // 1D Sobel kernels
        Mat kernelX = new Mat(3,3, CvType.CV_64F)
        {
            {
                put(0,0,0);
                put(0,1,0);
                put(0,2,0);

                put(1,0,-1);
                put(1,1,0);
                put(1,2,1);

                put(2,0,0);
                put(2,1,0);
                put(2,2,0);
            }
        };
        Mat kernelY = new Mat(3,3, CvType.CV_64F)
        {
            {
                put(0,0,0);
                put(0,1,-1);
                put(0,2,0);

                put(1,0,0);
                put(1,1,0);
                put(1,2,0);

                put(2,0,0);
                put(2,1,1);
                put(2,2,0);
            }
        };
        for (Mat colorChannel : channels) {
            Mat gradientXMat = new Mat(colorChannel.rows(), colorChannel.cols(), colorChannel.type());
            Mat gradientYMat = new Mat(colorChannel.rows(), colorChannel.cols(), colorChannel.type());
            Imgproc.filter2D(colorChannel, gradientXMat, -1, kernelX);
            Imgproc.filter2D(colorChannel, gradientYMat, -1, kernelY);
//            System.out.println(Utils.testHorizontalGradient(colorChannel, gradientXMat));
//            System.out.println(Utils.testVerticalGradient(colorChannel, gradientYMat));
//            Mat gradientXMat8UC = Utils.convertGradientImageToGrayscale(gradientXMat);
//            Mat gradientYMat8UC = Utils.convertGradientImageToGrayscale(gradientYMat);
//            Imgcodecs.imwrite(DEBUGPATH + "gradX_"+i+".png", gradientXMat8UC);
//            Imgcodecs.imwrite(DEBUGPATH + "gradY_"+i+".png", gradientYMat8UC);
//            Utils.showImageInPopup(Utils.matToBufferedImage(gradientXMat8UC, null));
//            Utils.showImageInPopup(Utils.matToBufferedImage(gradientYMat8UC, null));
            grads[0].add(gradientXMat);
            grads[1].add(gradientYMat);
        }

        // Get gradient magnitude and angles, for every channel
        List<Mat> magnitudes = new ArrayList<>();
        List<Mat> angles = new ArrayList<>();
        for(int channel = 0; channel < img.channels(); channel++)
        {
//            System.out.println("dx:"+grads[0].get(channel).get(7,7)[0]);
//            System.out.println("dy:"+grads[1].get(channel).get(7,7)[0]);
            Mat gradX = grads[0].get(channel);
            Mat gradY = grads[1].get(channel);
            Mat magnitude = new Mat();
            Mat angle = new Mat();
            Core.cartToPolar(gradX, gradY, magnitude, angle, true);
            //Zero - out the edges. They are unreliably noisy due to the extrapolation of the filter2D method.
            for(int i=0;i<magnitude.rows();i++)
            {
                for(int j=0;j<magnitude.cols();j++)
                {
                    if(i==0 || j==0 || i==magnitude.rows()-1 || j==magnitude.cols()-1)
                    {
                        magnitude.put(i,j,0);
                        angle.put(i,j,0);
                    }
                }
            }
//            System.out.println(magnitude.dump());
//            System.out.println(angle.dump());
            magnitudes.add(magnitude);
            angles.add(angle);
        }
        // Pick the final gradient as the gradient of the channel with the largest magnitude
        Mat finalMagnitude = new Mat(img.rows(), img.cols(), CvType.CV_64F);
        Mat finalAngle = new Mat(img.rows(), img.cols(), CvType.CV_64F);
        for(int i=0;i<img.rows();i++)
        {
            for(int j=0;j<img.cols();j++)
            {
                double maxMagnitude = -Double.MAX_VALUE;
                int selectedChannel = -1;
                for(int channel = 0; channel < img.channels(); channel++)
                {
                    double channelMagnitude = magnitudes.get(channel).get(i,j)[0];
//                    if(i == 7 && j == 7)
//                        System.out.println(channelMagnitude);
                    if(channelMagnitude > maxMagnitude)
                    {
                        maxMagnitude = channelMagnitude;
                        selectedChannel = channel;
                    }
                }
                double selectedMagnitude = magnitudes.get(selectedChannel).get(i,j)[0];
                double selectedAngle = angles.get(selectedChannel).get(i,j)[0];
                finalMagnitude.put(i,j,selectedMagnitude);
                finalAngle.put(i,j,selectedAngle);
            }
        }
        return new Mat[]{finalMagnitude, finalAngle};
    }

    private static Table<Integer, Integer, Mat> getPatchTable(Mat magnitudes, Mat angles)
    {
        int verticalPatchCount = angles.rows() / PATCH_SIZE;
        int horizontalPatchCount = angles.cols() / PATCH_SIZE;
        Table<Integer, Integer, Mat> histogramTable = HashBasedTable.create();
        for(int u=0;u<verticalPatchCount;u++)
        {
            for(int v=0;v<horizontalPatchCount;v++)
            {
                // Mat histogram = Mat.zeros(1, BIN_COUNT, CvType.CV_64F);
                double [] histogram = new double[BIN_COUNT];
                Arrays.fill(histogram, 0.0);
                double binLength = 180.0 / BIN_COUNT;
                int topLeftI = u*PATCH_SIZE;
                int topLeftJ = v*PATCH_SIZE;
                for(int i=0;i<PATCH_SIZE;i++)
                {
                    for (int j=0;j<PATCH_SIZE;j++)
                    {
                        int I = topLeftI + i;
                        int J = topLeftJ + j;
                        double magnitude = magnitudes.get(I,J)[0];
                        double angle = angles.get(I,J)[0];
                        double unsignedAngle = angle % 180.0;
                        int lowerBinIndex = ((int)Math.floor(unsignedAngle / binLength)) % BIN_COUNT;
                        int upperBinIndex = (lowerBinIndex + 1) % BIN_COUNT;
                        // Calculate the proportions to go into both bins.
                        double lowerBound = (double)lowerBinIndex * binLength;
                        double upperBound = (double)upperBinIndex * binLength;
                        if(upperBound < unsignedAngle)
                            upperBound += 180.0;
                        assert (lowerBound <= unsignedAngle) && (unsignedAngle <= upperBound);
                        double lowerBinWeight = (upperBound - unsignedAngle) / binLength;
                        double upperBinWeight = (unsignedAngle - lowerBound) / binLength;
//                        System.out.println("("+I+","+J+")");
//                        System.out.println("Mag:"+magnitude);
//                        System.out.println("Ang:"+angle);
//                        System.out.println("lowerBinIndex:"+lowerBinIndex);
//                        System.out.println("upperBinIndex:"+upperBinIndex);
//                        System.out.println("lowerBinWeight:"+lowerBinWeight);
//                        System.out.println("upperBinWeight:"+upperBinWeight);
                        histogram[lowerBinIndex] += lowerBinWeight*magnitude;
                        histogram[upperBinIndex] += upperBinWeight*magnitude;
                    }
                }
                Mat histogramAsMat = new Mat(BIN_COUNT, 1, CvType.CV_64F);
                for(int i=0;i<BIN_COUNT;i++)
                    histogramAsMat.put(i,0,histogram[i]);
                // System.out.println(histogramAsMat.dump());
                histogramTable.put(u,v,histogramAsMat);
            }
        }
        return histogramTable;
    }

    private static Mat normalizeHistograms(Table<Integer, Integer, Mat> histogramTable, int width, int height)
    {
        int verticalPatchCount = height / PATCH_SIZE;
        int horizontalPatchCount = width / PATCH_SIZE;
        Mat finalFeatureVector = new Mat();
        List<Mat> listOfNormalizedVectors = new ArrayList<>();
        for(int i=0;i<verticalPatchCount-1;i++)
        {
            for(int j=0;j<horizontalPatchCount-1;j++)
            {
                Mat topLeftHistogram = histogramTable.get(i,j);
                Mat topRightHistogram = histogramTable.get(i,j+1);
                Mat bottomLeftHistogram = histogramTable.get(i+1,j);
                Mat bottomRightHistogram = histogramTable.get(i+1,j+1);
                List<Mat> histogramList = new ArrayList<>(Arrays.asList(topLeftHistogram, topRightHistogram,
                        bottomLeftHistogram, bottomRightHistogram));
                Mat concatenatedHistogram = new Mat();
                Mat normalizedHistogram = new Mat();
//                System.out.println(topLeftHistogram.dump());
//                System.out.println(topRightHistogram.dump());
//                System.out.println(bottomLeftHistogram.dump());
//                System.out.println(bottomRightHistogram.dump());
                Core.vconcat(histogramList, concatenatedHistogram);
                //System.out.println(concatenatedHistogram.dump());
                Core.normalize(concatenatedHistogram, normalizedHistogram);
                //System.out.println(normalizedHistogram.dump());
                listOfNormalizedVectors.add(normalizedHistogram);
            }
        }
        Core.vconcat(listOfNormalizedVectors, finalFeatureVector);
        return finalFeatureVector;
    }

    public static Mat extractHOGFeature(Mat img) throws Exception {
        // Step check if it has correct sizes.
        if(img.cols()%PATCH_SIZE != 0 || img.rows()%PATCH_SIZE != 0)
        {
            System.out.println("Invalid image size");
            throw new Exception("Invalid image size");
        }
        // Get gradients
        Mat[] magnitudeAndAngles = getGradients(img);
        // Get patch-wise gradient histograms, in a table.
        Table<Integer, Integer, Mat> histogramTable = getPatchTable(magnitudeAndAngles[0], magnitudeAndAngles[1]);
        // Apply block normalization and get the final HOG feature vector.
        Mat hogFeature = normalizeHistograms(histogramTable, magnitudeAndAngles[0].cols(), magnitudeAndAngles[1].rows());
        return hogFeature;
    }

    public static Mat extractOpenCVHogFeature(Mat img, int patch_width, int patch_height) {
        Size windowSize = new Size(patch_width, patch_height);
        Size blockSize = new Size(BLOCK_SIZE, BLOCK_SIZE);
        Size blockStride = new Size(PATCH_SIZE,PATCH_SIZE);
        Size cellSize = new Size(PATCH_SIZE,PATCH_SIZE);
        int nbins = 9;
        // Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize, int _nbins
        HOGDescriptor hogDescriptor = new HOGDescriptor(windowSize, blockSize, blockStride, cellSize, nbins);
        MatOfFloat descriptors = new MatOfFloat();
        hogDescriptor.compute(img, descriptors);
        return descriptors;
    }


}
