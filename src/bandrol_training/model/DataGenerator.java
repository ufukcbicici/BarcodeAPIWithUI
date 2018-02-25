package bandrol_training.model;

import bandrol_training.Constants;
import com.google.common.collect.Sets;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static bandrol_training.Constants.LOCALIZED_IMAGE_PATH;

public class DataGenerator {

    public static List<Mat> augmentSample(String fileName,
                                     Mat sourceImage,
                                     GroundTruth groundTruth,
                                     double minRotationAngle,
                                     double stepRotationAngle,
                                     double maxRotationAngle,
                                     double minHorizontalOffset,
                                     double stepHorizontal,
                                     double maxHorizontalOffset,
                                     double minVerticalOffset,
                                     double stepVertical,
                                     double maxVerticalOffset)
    {
        Point imageCenter =
                new Point((double)groundTruth.x + (double)groundTruth.width / 2.0,
                          (double)groundTruth.y + (double)groundTruth.height / 2.0);
        Set<Double> rotationAngles = new HashSet<>(Utils.rangeToList(minRotationAngle, maxRotationAngle, stepRotationAngle));
        Set<Double> horizontalOffsets = new HashSet<>(Utils.rangeToList(minHorizontalOffset, maxHorizontalOffset,
                stepHorizontal));
        Set<Double> verticalOffsets = new HashSet<>(Utils.rangeToList(minVerticalOffset, maxVerticalOffset, stepVertical));
        Set<List<Double>> cartesianProduct = Sets.cartesianProduct(rotationAngles, horizontalOffsets, verticalOffsets);
        List<Mat> augmentedSamples = new ArrayList<>();
        for(List<Double> transformation : cartesianProduct)
        {
            double rotationAngle = transformation.get(0);
            double horizontalOffset = transformation.get(1);
            double verticalOffset = transformation.get(2);
            Mat rotationMatrix = Imgproc.getRotationMatrix2D(imageCenter, rotationAngle,1.0);
            Mat rotatedImg = sourceImage.clone();
            rotatedImg.setTo(Constants.CHROME_GREEN);
            Imgproc.warpAffine(sourceImage, rotatedImg, rotationMatrix, new Size(sourceImage.cols(),
                    sourceImage.rows()));
            double translation [] = {  1, 0, horizontalOffset, 0, 1, verticalOffset };
            Mat translationMatrix = new Mat(2, 3, CvType.CV_64F);
            translationMatrix.put(0,0, translation);
            // System.out.println(translationMatrix.dump());
            Mat translatedImg = sourceImage.clone();
            translatedImg.setTo(Constants.CHROME_GREEN);
            Imgproc.warpAffine(rotatedImg, translatedImg, translationMatrix, new Size(rotatedImg.cols(),
                    rotatedImg.rows()));
            Mat augmentedSample = translatedImg.submat(groundTruth.y, groundTruth.y + groundTruth.height,
                    groundTruth.x, groundTruth.x + groundTruth.width);
            // Check if it includes borders
            Mat thresholdedSample = new Mat();
            Core.inRange(augmentedSample, new Scalar(0,0,0), new Scalar(0,0,0), thresholdedSample);
            Scalar sumResult = Core.sumElems(thresholdedSample);
//            System.out.println(augmentedSample.cols());
//            System.out.println(augmentedSample.rows());
//            Utils.showImageInPopup(Utils.matToBufferedImage(augmentedSample, null));
            if(sumResult.val[0] + sumResult.val[1] + sumResult.val[2] > 0)
                continue;
            String sampleName = "positiveSample_"+fileName+"_"+groundTruth.label+"_"+"("+groundTruth.x+","+groundTruth.y+")_("
                    +rotationAngle+","+horizontalOffset+","+verticalOffset+")";
            String suitableFileName = Utils.getNonExistingFileName(LOCALIZED_IMAGE_PATH + sampleName,
                    ".png");
            Imgcodecs.imwrite(suitableFileName, augmentedSample);
            augmentedSamples.add(augmentedSample);
        }
        return augmentedSamples;
    }

}
