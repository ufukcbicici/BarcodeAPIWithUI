package bandrol_training.model;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class HOGExtractor
{
    private static final int PATCH_SIZE = 8;
    private static final int BLOCK_SIZE = 16;

    private static List<Mat> [] getGradients(Mat img)
    {
        // Step 1: Split into RGB channels.
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels);

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
        for(Mat colorChannel : channels)
        {
            Mat gradientXMat = new Mat(colorChannel.rows(), colorChannel.cols(), colorChannel.type());
            Mat gradientYMat = new Mat(colorChannel.rows(), colorChannel.cols(), colorChannel.type());
            Imgproc.filter2D(colorChannel, gradientXMat, -1, kernelX);
            Imgproc.filter2D(colorChannel, gradientYMat, -1, kernelX);
            grads[0].add(gradientXMat);
            grads[1].add(gradientYMat);
        }
        return grads;
    }

    public static Mat extractHOGFeature(Mat img) throws Exception {
        // Step check if it has correct sizes.
        if(img.cols()%PATCH_SIZE != 0 || img.rows()%PATCH_SIZE != 0)
        {
            System.out.println("Invalid image size");
            throw new Exception("Invalid image size");
        }
        // Get gradients
        List<Mat> [] grads = getGradients(img);
        //

        return new Mat();

    }


}
