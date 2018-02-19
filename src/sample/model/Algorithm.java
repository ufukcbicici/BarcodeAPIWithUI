package sample.model;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import sample.model.OrientationFinding.OrientationFinder;
import sample.model.QRCodeReading.IQRCodeReader;
import sample.model.QRCodeReading.QRCodePoint;
import sample.model.QRCodeReading.QRCodeReaders.ZxingQRCodeReader;
import sample.model.QRCodeReading.PipelineInfo;

import java.awt.image.BufferedImage;
import java.util.List;

import static sample.Constants.DEBUGPATH;

public class Algorithm {
    private static final Point finder0 = new Point(1616, 1211);
    private static final Point finder1 = new Point(1939, 1207);
    private static final Point edge0 = new Point(1534, 1215);
    private static final Point edge1 = new Point(2011,1209);

    private static double getEdgeToFinderDistanceRatio()
    {
        double distance0 = Utils.getDistanceBetweenPoints(finder0, finder1);
        double distance1 = Utils.getDistanceBetweenPoints(edge0, edge1);
        return (distance1 - distance0) / (2.0 * distance0);
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
        for(QRCodePoint finderPattern : pipelineInfo.getFinderPatterns())
            Imgproc.circle(sourceImg, finderPattern.getLocation(), 5, new Scalar(255, 0,0 ), -1);
        // Draw alignment pattern
        Imgproc.circle(sourceImg, pipelineInfo.getAlignmentPattern().getLocation(), 5, new Scalar(255, 0,0 ), -1);
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
        Imgproc.circle(rotatedImg, new Point(bottomLeftPoint.get(0,0)[0], bottomLeftPoint.get(1,0)[0]),
                5, new Scalar(0, 255,0 ), -1);
        Imgproc.circle(rotatedImg, new Point(topRightPoint.get(0,0)[0], topRightPoint.get(1,0)[0]),
                5, new Scalar(0, 255,0 ), -1);
        Imgproc.circle(rotatedImg, new Point(bottomRightPoint.get(0,0)[0], bottomRightPoint.get(1,0)[0]),
                5, new Scalar(0, 255,0 ), -1);
        // Draw a line through rotated finder points
        Utils.drawLineOnMat(rotatedImg, bottomLeftPoint, bottomRightPoint, new Scalar(255,255,255), 10);
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
        Imgproc.circle(rotatedImg, bottom_middle_point,5, new Scalar(255, 255,0 ), -1);
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
        Utils.drawLineOnMat(rotatedImg, middleBottom, bottom_middle_displaced_mat, new Scalar(255,255,255), 10);



        // ****************************** Part 2 ******************************
        // Draw Image
        Imgcodecs.imwrite(DEBUGPATH + "rotated.png", rotatedImg);
        Mat resizedImg = new Mat();
        Imgproc.resize(rotatedImg, resizedImg, new Size(0.3*sourceImg.width(), 0.3*sourceImg.height()));
        Utils.showImageInPopup(Utils.matToBufferedImage(resizedImg, null));
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
    }
}
