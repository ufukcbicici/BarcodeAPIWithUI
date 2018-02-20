package bandrol_training.model;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import bandrol_training.model.OrientationFinding.OrientationFinder;
import bandrol_training.model.QRCodeReading.IQRCodeReader;
import bandrol_training.model.QRCodeReading.QRCodePoint;
import bandrol_training.model.QRCodeReading.QRCodeReaders.ZxingQRCodeReader;
import bandrol_training.model.QRCodeReading.PipelineInfo;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static bandrol_training.Constants.DEBUGPATH;
import static bandrol_training.Constants.DOWNSIZE_RATIO;

public class Algorithm {
    private static final Point finder0 = new Point(1616, 1211);
    private static final Point finder1 = new Point(1939, 1207);
    private static final Point edge0 = new Point(1534, 1215);
    private static final Point edge1 = new Point(2011,1209);

    private static final Point corner0 = new Point(2005,884);
    private static final Point corner1 = new Point(2010,1136);
    // private static final Point upperLimit = new Point(2005,880);

    private static double getEdgeToFinderDistanceRatio()
    {
        double distance0 = Utils.getDistanceBetweenPoints(finder0, finder1);
        double distance1 = Utils.getDistanceBetweenPoints(edge0, edge1);
        return (distance1 - distance0) / (2.0 * distance0);
    }

    private static double getLookUpWindowHeightRatio()
    {
        double distance0 = Utils.getDistanceBetweenPoints(finder0, finder1);
        double distance1 = Utils.getDistanceBetweenPoints(corner0, corner1);
        return distance1 / distance0;
    }

    public static IQRCodeReader qrCodeReader;

    public static void InitAlgorithm()
    {
        qrCodeReader = new ZxingQRCodeReader();
    }

    public static List<QRCodePoint> getQRCodePoints(PipelineInfo pipelineInfo, List<Point> corners)
    {
        return qrCodeReader.detectQRCode(pipelineInfo.getOriginalImage(), corners);
    }

    public static PipelineInfo getOrientationFromQRCodePoints(PipelineInfo pipelineInfo,
                                                              List<QRCodePoint> qrCodePointList)
    {
        return OrientationFinder.findOrientationFromQRCode(pipelineInfo, qrCodePointList);
    }

    private static void localizeSerialNumber(PipelineInfo pipelineInfo) {
        // ****************************** Part 1 ******************************
        Mat sourceImg = Utils.bufferedImageToMat(pipelineInfo.getOriginalImage());
        // Draw finder patterns
//        for(QRCodePoint finderPattern : pipelineInfo.getFinderPatterns())
//            Imgproc.circle(sourceImg, finderPattern.getLocation(), 5, new Scalar(255, 0,0 ), -1);
        // Draw alignment pattern
        // Imgproc.circle(sourceImg, pipelineInfo.getAlignmentPattern().getLocation(), 5, new Scalar(255, 0,0 ), -1);
        // Draw up vector
        Mat upVecAsMat = pipelineInfo.getRotationUpVectorStartFinishPoints();
        Mat up0 = new Mat();
        Mat up1 = new Mat();
        Core.transpose(upVecAsMat.row(0), up0);
        Core.transpose(upVecAsMat.row(1), up1);
        Utils.drawLineOnMat(sourceImg, up0, up1, new Scalar(0,255,0), 15);
        // Rotate the image
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(pipelineInfo.getRotationCenter(), pipelineInfo.getAngle(),
                1.0);
        Mat rotatedImg = new Mat();
        Imgproc.warpAffine(sourceImg, rotatedImg, rotationMatrix, new Size(sourceImg.cols(), sourceImg.rows()));
        // Localize the search area for the serial number.
        // Determine rotated finder patterns as bottom-left, bottom-right and top-right ones, after rotation.
        Mat A = new Mat(2, 1, CvType.CV_64F);
        Mat B = new Mat(2, 1, CvType.CV_64F);
        Mat C = new Mat(2, 1, CvType.CV_64F);
        A.put(0,0,pipelineInfo.getFinderPatterns().get(0).getLocation().x);
        A.put(1,0,pipelineInfo.getFinderPatterns().get(0).getLocation().y);
        B.put(0,0,pipelineInfo.getFinderPatterns().get(1).getLocation().x);
        B.put(1,0,pipelineInfo.getFinderPatterns().get(1).getLocation().y);
        C.put(0,0,pipelineInfo.getFinderPatterns().get(2).getLocation().x);
        C.put(1,0,pipelineInfo.getFinderPatterns().get(2).getLocation().y);
        Mat [] points = {A,B,C};
        Mat [] rotatedPoints = new Mat[3];
        Mat topRightPoint = new Mat();
        Mat bottomLeftPoint = new Mat();
        Mat bottomRightPoint = new Mat();
        double minY = Double.MAX_VALUE;
        double minX = Double.MAX_VALUE;
        for(int i=0;i<points.length;i++)
        {
            Mat rotatedPoint;
            try {
                rotatedPoint = Utils.rotate2DPoint(points[i], pipelineInfo.getRotationCenter(), pipelineInfo.getAngle());
                rotatedPoints[i] = rotatedPoint;
                if(rotatedPoint.get(0,0)[0] < minX)
                {
                    minX = rotatedPoint.get(0,0)[0];
                    bottomLeftPoint = rotatedPoint;
                }
                if(rotatedPoint.get(1,0)[0] < minY)
                {
                    minY = rotatedPoint.get(1,0)[0];
                    topRightPoint = rotatedPoint;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        for (Mat rotatedPoint : rotatedPoints) {
            if (rotatedPoint != bottomLeftPoint && rotatedPoint != topRightPoint) {
                bottomRightPoint = rotatedPoint;
                break;
            }
        }
        // Draw rotated finder patterns.
//        Imgproc.circle(rotatedImg, new Point(bottomLeftPoint.get(0,0)[0], bottomLeftPoint.get(1,0)[0]),
//                5, new Scalar(0, 255,0 ), -1);
//        Imgproc.circle(rotatedImg, new Point(topRightPoint.get(0,0)[0], topRightPoint.get(1,0)[0]),
//                5, new Scalar(0, 255,0 ), -1);
//        Imgproc.circle(rotatedImg, new Point(bottomRightPoint.get(0,0)[0], bottomRightPoint.get(1,0)[0]),
//                5, new Scalar(0, 255,0 ), -1);
        // Draw a line through rotated finder points
        // Utils.drawLineOnMat(rotatedImg, bottomLeftPoint, bottomRightPoint, new Scalar(255,255,255), 10);
        // ****************************** Part 1 ******************************

        // ****************************** Part 2 ******************************
        // Calculate the vector: bottom_v = (bottomLeftPoint - bottomRightPoint)
        Mat bottom_v = new Mat();
        Core.subtract(bottomLeftPoint, bottomRightPoint, bottom_v);
        // Find the middle point
        Mat half_bottom_v = new Mat();
        Mat middleBottom = new Mat();
        Core.multiply(bottom_v, new Scalar(0.5), half_bottom_v);
        Core.add(bottomRightPoint, half_bottom_v, middleBottom);
        // Draw middle point
        Point bottom_middle_point = Utils.convertColumnMatTo2DPoint(middleBottom);
        // Imgproc.circle(rotatedImg, bottom_middle_point,5, new Scalar(255, 255,0 ), -1);
        // Calculate the normal to bottom_v
        Mat bottom_v_normalized = new Mat();
        Core.normalize(bottom_v, bottom_v_normalized);
        Mat normal_to_bottom_v = bottom_v.clone();
        //(x0,y0) = bottomLeftPoint, (x1,y1) = bottomRightPoint
        // (A,B) = (y0 - y1; x1 - x0).
        normal_to_bottom_v.put(
                0,0, bottomRightPoint.get(1,0)[0] - bottomLeftPoint.get(1,0)[0]);
        normal_to_bottom_v.put(
                1,0, bottomLeftPoint.get(0,0)[0] - bottomRightPoint.get(0,0)[0]);
        Core.normalize(normal_to_bottom_v.clone(), normal_to_bottom_v);
        double half_bottom_v_length = Core.norm(half_bottom_v);
        Mat scaled_normal = new Mat();
        Core.multiply(normal_to_bottom_v, new Scalar(half_bottom_v_length), scaled_normal);
        Mat bottom_middle_displaced_mat = new Mat();
        Core.add(middleBottom, scaled_normal, bottom_middle_displaced_mat);
        // Draw normal line
        // Utils.drawLineOnMat(rotatedImg, middleBottom, bottom_middle_displaced_mat, new Scalar(255,255,255), 10);
        //Get the angle between the normal_to_bottom_v and up vector.
        Mat upVector = new Mat(2,1,CvType.CV_64F);
        upVector.put(0,0, 0);
        upVector.put(1,0, -1);
        System.out.println(Core.norm(normal_to_bottom_v));
        double cos_correction_angle = upVector.dot(normal_to_bottom_v);
        double correction_angle = Math.acos(cos_correction_angle) * (180.0 / Math.PI);
        System.out.println(correction_angle);
        if(normal_to_bottom_v.get(0,0)[0] < 0.0)
            correction_angle *= -1.0;
        //Rotate the image, for the last time.
        Mat correctionRotMatrix = Imgproc.getRotationMatrix2D(bottom_middle_point, correction_angle,1.0);
        Mat correctedImg = new Mat();
        Imgproc.warpAffine(rotatedImg, correctedImg, correctionRotMatrix, new Size(rotatedImg.cols(), rotatedImg.rows()));
        //Rotate the finder patterns
        Mat topRightPointCorrected = new Mat();
        Mat bottomLeftPointCorrected = new Mat();
        Mat bottomRightPointCorrected = new Mat();
        try {
            topRightPointCorrected = Utils.rotate2DPoint(topRightPoint, bottom_middle_point, correction_angle);
            bottomLeftPointCorrected = Utils.rotate2DPoint(bottomLeftPoint, bottom_middle_point, correction_angle);
            bottomRightPointCorrected = Utils.rotate2DPoint(bottomRightPoint, bottom_middle_point, correction_angle);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Draw rotated finder patterns.
//        Imgproc.circle(correctedImg, new Point(bottomLeftPointCorrected.get(0,0)[0], bottomLeftPointCorrected.get(1,0)[0]),
//                10, new Scalar(255, 128,0 ), -1);
//        Imgproc.circle(correctedImg, new Point(topRightPointCorrected.get(0,0)[0], topRightPointCorrected.get(1,0)[0]),
//                10, new Scalar(255, 128,0 ), -1);
//        Imgproc.circle(correctedImg, new Point(bottomRightPointCorrected.get(0,0)[0], bottomRightPointCorrected.get(1,0)[0]),
//                10, new Scalar(255, 128,0 ), -1);
        // ****************************** Part 2 ******************************

        // ****************************** Part 3 ******************************
        // Localize the serial number
        Core.subtract(bottomLeftPointCorrected, bottomRightPointCorrected, bottom_v);
        normal_to_bottom_v = bottom_v.clone();
        //(x0,y0) = bottomLeftPoint, (x1,y1) = bottomRightPoint
        // (A,B) = (y0 - y1; x1 - x0).
        normal_to_bottom_v.put(
                0,0, bottomLeftPointCorrected.get(1,0)[0] - bottomRightPointCorrected.get(1,0)[0]);
        normal_to_bottom_v.put(
                1,0, bottomRightPointCorrected.get(0,0)[0] - bottomLeftPointCorrected.get(0,0)[0]);
        double outlierRatio = getEdgeToFinderDistanceRatio();
        double bottomFindersDistance = Core.norm(bottom_v);
        Core.normalize(bottom_v.clone(), bottom_v);
        Core.normalize(normal_to_bottom_v.clone(), normal_to_bottom_v);
        Mat serialNoBoundingBoxTopLeft = new Mat();
        Mat serialNoBoundingBoxTopRight = new Mat();
        Mat serialNoBoundingBoxBottomLeft = new Mat();
        Mat serialNoBoundingBoxBottomRight = new Mat();
        Mat displacementHorizontal = new Mat();
        Mat displacementVertical = new Mat();
        Mat totalDisplacement = new Mat();
        // Serial number bounding box top left corner
        Core.multiply(bottom_v, new Scalar(outlierRatio * bottomFindersDistance), displacementHorizontal);
        Core.multiply(normal_to_bottom_v, new Scalar(outlierRatio * bottomFindersDistance), displacementVertical);
        // Core.add(displacementHorizontal, displacementVertical, totalDisplacement);
        Core.add(bottomLeftPointCorrected, displacementHorizontal, serialNoBoundingBoxTopLeft);
        System.out.println("bottomLeftPointCorrected");
        System.out.println(bottomLeftPointCorrected.dump());
        System.out.println("serialNoBoundingBoxTopLeft");
        System.out.println(serialNoBoundingBoxTopLeft.dump());
        // Serial number bounding box top right corner
        Core.multiply(bottom_v, new Scalar(-outlierRatio * bottomFindersDistance), displacementHorizontal);
        // Core.add(displacementHorizontal, displacementVertical, totalDisplacement);
        Core.add(bottomRightPointCorrected, displacementHorizontal, serialNoBoundingBoxTopRight);
        // Serial number bounding box bottom left/right corners
        double boundingBoxHeightToBottomFindersDistanceRatio = getLookUpWindowHeightRatio();
        Core.multiply(normal_to_bottom_v,
                new Scalar((boundingBoxHeightToBottomFindersDistanceRatio + outlierRatio) * bottomFindersDistance),
                displacementVertical);
        Core.add(serialNoBoundingBoxTopLeft, displacementVertical, serialNoBoundingBoxBottomLeft);
        Core.add(serialNoBoundingBoxTopRight, displacementVertical, serialNoBoundingBoxBottomRight);
        List<Point> cornerPoints = new ArrayList<>();
        cornerPoints.add(Utils.convertColumnMatTo2DPoint(serialNoBoundingBoxTopLeft));
        cornerPoints.add(Utils.convertColumnMatTo2DPoint(serialNoBoundingBoxTopRight));
        cornerPoints.add(Utils.convertColumnMatTo2DPoint(serialNoBoundingBoxBottomLeft));
        cornerPoints.add(Utils.convertColumnMatTo2DPoint(serialNoBoundingBoxBottomRight));
        Rect2d boundingBox = Utils.getTightestBoundingRectangle(cornerPoints);
        pipelineInfo.setLocalizedSerialNumberImg(correctedImg.submat((int)boundingBox.y, (int)(boundingBox.y + boundingBox.height),
                (int)boundingBox.x, (int)(boundingBox.x + boundingBox.width)).clone());
        // Draw the bounding box corners
//        Imgproc.circle(correctedImg, new Point(serialNoBoundingBoxTopLeft.get(0,0)[0], serialNoBoundingBoxTopLeft.get(1,0)[0]),
//                10, new Scalar(0, 255, 0 ), -1);
//        Imgproc.circle(correctedImg, new Point(serialNoBoundingBoxTopRight.get(0,0)[0], serialNoBoundingBoxTopRight.get(1,0)[0]),
//                10, new Scalar(0, 255, 0 ), -1);
//        Imgproc.circle(correctedImg, new Point(serialNoBoundingBoxBottomLeft.get(0,0)[0], serialNoBoundingBoxBottomLeft.get(1,0)[0]),
//                10, new Scalar(0, 255, 0 ), -1);
//        Imgproc.circle(correctedImg, new Point(serialNoBoundingBoxBottomRight.get(0,0)[0], serialNoBoundingBoxBottomRight.get(1,0)[0]),
//                10, new Scalar(0, 255, 0 ), -1);
        // Draw the bounding box edges
//        Utils.drawLineOnMat(correctedImg, serialNoBoundingBoxTopLeft, serialNoBoundingBoxTopRight,
//                new Scalar(0,255,0), 10);
//        Utils.drawLineOnMat(correctedImg, serialNoBoundingBoxTopRight, serialNoBoundingBoxBottomRight,
//                new Scalar(0,255,0), 10);
//        Utils.drawLineOnMat(correctedImg, serialNoBoundingBoxBottomRight, serialNoBoundingBoxBottomLeft,
//                new Scalar(0,255,0), 10);
//        Utils.drawLineOnMat(correctedImg, serialNoBoundingBoxBottomLeft, serialNoBoundingBoxTopLeft,
//                new Scalar(0,255,0), 10);
        // ****************************** Part 3 ******************************

        // Draw Image
        Imgcodecs.imwrite(DEBUGPATH + "rotated_corrected.png", correctedImg);
        Mat resizedImg = new Mat();
        Imgproc.resize(correctedImg, resizedImg,
                new Size(0.2*correctedImg.width(),0.2*correctedImg.height()));
        Utils.showImageInPopup(Utils.matToBufferedImage(resizedImg, null));
    }

    private static void localizeCharactersUsingColor(PipelineInfo pipelineInfo)
    {
        Mat originalImage = pipelineInfo.getLocalizedSerialNumberImg();
        String suitableFileName = Utils.getNonExistingFileName(DEBUGPATH + "localized", ".png");
        Imgcodecs.imwrite(suitableFileName, originalImage);
        Utils.showImageInPopup(Utils.matToBufferedImage(originalImage, null));
        Mat downsized = new Mat();
        Imgproc.resize(originalImage, downsized,
                new Size(DOWNSIZE_RATIO*originalImage.width(),DOWNSIZE_RATIO*originalImage.height()));
        // Apply two k=2 k-means.
        Map<Mat, Integer> clusterMap = Cluster.cluster(downsized, 2);
        // The cluster with larger number of count should be the background.
        Mat background = null;
        Mat foreground = null;
        List<Mat> clusters = new ArrayList<>(clusterMap.keySet());
        if(clusterMap.get(clusters.get(0)) > clusterMap.get(clusters.get(1)))
        {
            background = clusters.get(0);
            foreground = clusters.get(1);
        }
        else
        {
            background = clusters.get(1);
            foreground = clusters.get(0);
        }
        Utils.showImageInPopup(Utils.matToBufferedImage(foreground, null));


    }

    private static void localizeCharacters(PipelineInfo pipelineInfo)
    {
        Utils.showImageInPopup(Utils.matToBufferedImage(pipelineInfo.getLocalizedSerialNumberImg(), null));
        // Convert to grayscale
        Mat grayScaleImg = new Mat();
        Imgproc.cvtColor(pipelineInfo.getLocalizedSerialNumberImg(), grayScaleImg, Imgproc.COLOR_RGB2GRAY);
        Mat downsized = new Mat();
        Imgproc.resize(grayScaleImg, downsized,
                new Size(0.5*grayScaleImg.width(),0.5*grayScaleImg.height()));
        grayScaleImg = downsized;
        Utils.showImageInPopup(Utils.matToBufferedImage(grayScaleImg, null));
        double [] filterSizes = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31};
        for(int i = 0;i < filterSizes.length;i++)
        {
            Mat blurredImg = new Mat();
            double filterSize = filterSizes[i];
//            Imgproc.GaussianBlur(grayScaleImg, blurredImg, new Size(filterSize,filterSize),
//                    0);
            Imgproc.medianBlur(grayScaleImg, blurredImg, (int)filterSize);
            Imgcodecs.imwrite(DEBUGPATH + "gray_scale_serial_"+"median"+(int)filterSize+".png", blurredImg);
            Mat thresholdedImg = new Mat();
            Imgproc.threshold(blurredImg, thresholdedImg, 0, 255,
                    Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
            Imgcodecs.imwrite(DEBUGPATH + "otsu_"+"blur"+(int)filterSize+".png", thresholdedImg);
            // Utils.showImageInPopup(Utils.matToBufferedImage(thresholdedImg, null));
        }



        // Gaussian Blur
//        Mat blurredImg = new Mat();
//        Imgproc.GaussianBlur(grayScaleImg, blurredImg, new Size(CHARACTER_LOC_GAUSSIAN_KERNEL,CHARACTER_LOC_GAUSSIAN_KERNEL),
//                0);
//        Utils.showImageInPopup(Utils.matToBufferedImage(blurredImg, null));
////        // Histogram Equalization
////        Mat histogramEqualized = new Mat();
////        Imgproc.equalizeHist(blurredImg, histogramEqualized);
////        Utils.showImageInPopup(Utils.matToBufferedImage(histogramEqualized, null));
//        // Plot the histogram
//        Utils.showHistogram(blurredImg);
//        // Otsu Thresholding
//        Mat thresholdedImg = new Mat();
//        Imgproc.threshold(blurredImg, thresholdedImg, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
//        Utils.showImageInPopup(Utils.matToBufferedImage(thresholdedImg, null));

        // public static void calcHist(List<Mat> images, MatOfInt channels, Mat mask, Mat hist, MatOfInt histSize,
        // // MatOfFloat ranges, boolean accumulate)
//        Mat histogram = new Mat();
//        Mat normalizedHistogram = new Mat();
//        Imgproc.calcHist(
//                new ArrayList<Mat>(Arrays.asList(grayScaleImg)), new MatOfInt(0), new Mat(), histogram,
//                new MatOfInt(256), new MatOfFloat(0, 256), false);
//        Mat histogramImg = new Mat(400, 400, CvType.CV_8UC3, new Scalar(0,0,0));
//        // normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//        Core.normalize(histogram, normalizedHistogram, 0, histogramImg.rows(), Core.NORM_MINMAX);
//        int fhfh = 384834;

        // Histogram equalization
//        Mat histogramEqualized = new Mat();
//        Imgproc.equalizeHist(grayScaleImg, histogramEqualized);
//        Utils.showImageInPopup(Utils.matToBufferedImage(histogramEqualized, null));
    }

    public static void execute(BufferedImage img, List<Point> corners)
    {
        PipelineInfo pipelineInfo = new PipelineInfo();
        pipelineInfo.setOriginalImage(img);
        // Step 1: Detect the QR Code with Zxing library
        List<QRCodePoint> qrCodePointList = getQRCodePoints(pipelineInfo, corners);
        if (qrCodePointList == null)
            return;
        // Step 2: Get the QR Code orientation.
        getOrientationFromQRCodePoints(pipelineInfo, qrCodePointList);
        // Step 3: Rotate the image
        localizeSerialNumber(pipelineInfo);
        // Step 4: Localize characters
        localizeCharactersUsingColor(pipelineInfo);
        // localizeCharacters(pipelineInfo);
    }
}
