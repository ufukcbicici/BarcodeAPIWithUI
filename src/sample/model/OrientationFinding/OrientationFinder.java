package sample.model.OrientationFinding;

import org.opencv.core.*;
import sample.model.QRCodeReading.QRCodePoint;
import sample.model.QRCodeReading.QRCodePointTypes;
import sample.model.QRCodeReading.PipelineInfo;
import sample.model.Utils;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.stream.Collectors;

import static org.opencv.core.Core.dft;
import static org.opencv.core.Core.norm;

public class OrientationFinder {

    public static List<Point> getMostDistantFinderPoints(List<QRCodePoint> finderPatterns)
    {
        double maxDistance = Double.MIN_VALUE;
        QRCodePoint A = null;
        QRCodePoint B = null;
        QRCodePoint C = null;
        for(QRCodePoint p0 : finderPatterns)
        {
            for(QRCodePoint p1 : finderPatterns)
            {
                if(p0 == p1)
                    continue;
                double distance = Utils.getDistanceBetweenPoints(p0.getLocation(), p1.getLocation());
                if(distance > maxDistance)
                {
                    maxDistance = distance;
                    A = p0;
                    B = p1;
                    for(QRCodePoint p2 : finderPatterns)
                    {
                        if(p2 != p0 && p2 != p1)
                        {
                            C = p2;
                            break;
                        }
                    }
                }
            }
        }
        assert A != null;
        assert C != null;
        return new ArrayList<>(Arrays.asList(A.getLocation(), B.getLocation(), C.getLocation()));
    }

    private static Mat getMiddlePoint(Mat A, Mat B)
    {
        Mat diffVec = new Mat();
        Mat diffVecScaled = new Mat();
        Mat middlePoint = new Mat();
        Core.subtract(B, A, diffVec);
        Core.multiply(diffVec, new Scalar(0.5), diffVecScaled);
        Core.add(A, diffVecScaled, middlePoint);
        System.out.println(middlePoint.dump());
        return middlePoint;
    }

    private static Mat[] getReferenceVectors(Mat c_m_vector)
    {
        Mat[] referenceVectors = new Mat[4];
        // Obtain four reference vector by rotating diffVec by 45, 135, 225, 315 degrees.
        for(int i=0;i<4;i++)
        {
            double angle = 45.0 + (double)i * 90.0;
            Mat rotated;
            try {
                rotated = Utils.rotate2DPoint(c_m_vector, new Point(0.0, 0.0), angle);
                Core.normalize(rotated, rotated);
                System.out.println(rotated.size());
                System.out.println(rotated.dump());
                referenceVectors[i] = rotated;
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return referenceVectors;
    }

    private static void findUpVector(Mat c_m_vector, Mat[] referenceVectors, Mat middlePoint,
                                             List<QRCodePoint> QRCodePoints, PipelineInfo pipelineInfo) throws Exception {
        //Find the two vectors with which c_m_vector has the maximum angle.
        List<Mat> refList = new ArrayList<>(Arrays.asList(referenceVectors));
        refList.sort(Comparator.comparingDouble(v0 -> v0.dot(c_m_vector)));
        Mat v0 = refList.get(0);
        Mat v1 = refList.get(1);
        Point middle = null;
        try {
            middle = Utils.convertMatTo2DPoints(middlePoint).get(0);
        } catch (Exception e) {
            e.printStackTrace();
        }
        Mat [] candidateUpVectors = {v0, v1};
        for (Mat candUpVector : candidateUpVectors) {
            //Determine whether the rotation will clockwise or counter clockwise.
            //If x axis is negative, rotation will be clockwise. (Negative angle)
            //If x axis is positive, rotation will be counter clockwise. (Positive angle)
            //Determine the angle to the up vector.
            double cos_angle = candUpVector.dot(Utils.convert2DPointsToMat(Collections.singletonList(new Point(0.0, -1.0))));
            double angle = Math.acos(cos_angle) * (180.0 / Math.PI);
            if (candUpVector.get(0, 0)[0] <= 0.0)
                angle *= -1.0;
            //Rotate all QR Code Points
            List<Mat> rotatedFinderPoints = new ArrayList<>();
            Mat rotatedAlignmentPoint = null;
            for(QRCodePoint qrCodePoint : QRCodePoints)
            {
                Point loc = qrCodePoint.getLocation();
                Mat locAsMat = Utils.convert2DPointsToMat(Collections.singletonList(loc));
                Mat rotatedQRLoc = null;
                try {
                    rotatedQRLoc = Utils.rotate2DPoint(locAsMat, middle, angle);
                    if(qrCodePoint.getPointType() == QRCodePointTypes.FINDER_PATTERN)
                    {
                        rotatedFinderPoints.add(rotatedQRLoc);
                        System.out.println("*********");
                        System.out.println("Finder Pattern Before Rotation:("+locAsMat.get(0,0)[0]+","
                                +locAsMat.get(0,1)[0]+")");
                        System.out.println("Finder Pattern After Rotation:("+rotatedQRLoc.get(0,0)[0]+","
                                +rotatedQRLoc.get(0,1)[0]+")");
                        System.out.println("*********");
                    }
                    else
                    {
                        rotatedAlignmentPoint = rotatedQRLoc;
                        System.out.println("*********");
                        System.out.println("Alignment Pattern Before Rotation:("+locAsMat.get(0,0)[0]+","
                                +locAsMat.get(0,1)[0]+")");
                        System.out.println("Alignment Pattern After Rotation:("+rotatedQRLoc.get(0,0)[0]+","
                                +rotatedQRLoc.get(0,1)[0]+")");
                        System.out.println("*********");
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            //If the upper-left point is the alignment point after the rotation; then the current vector is the
            // up vector.
            Mat finalRotatedAlignmentPoint = rotatedAlignmentPoint;
            if (rotatedFinderPoints.stream().anyMatch(
                                (p) -> {
                                    assert finalRotatedAlignmentPoint != null;
                                    return finalRotatedAlignmentPoint.get(0, 0)[0] >= p.get(0, 0)[0] &&
                                            finalRotatedAlignmentPoint.get(0, 1)[0] >= p.get(0, 1)[0];
                                })) {
                continue;
            }
            pipelineInfo.setAngle(angle);
            pipelineInfo.setRotationCenter(middle);
            pipelineInfo.setRotationUpVector(candUpVector);
            return;
        }
        System.out.println("Failed to find the up vector!");
        throw new Exception("Failed to find the up vector!");
    }

    public static PipelineInfo findOrientationFromQRCode(PipelineInfo pipelineInfo, List<QRCodePoint> qrCodePointList)
    {
        List<QRCodePoint> finderPatterns = qrCodePointList.stream().
                filter((qrCodePoint -> qrCodePoint.getPointType() == QRCodePointTypes.FINDER_PATTERN)).
                collect(Collectors.toList());
        List<QRCodePoint> alignmentPatterns = qrCodePointList.stream().
                filter((qrCodePoint -> qrCodePoint.getPointType() == QRCodePointTypes.ALIGNMENT_PATTERN)).
                collect(Collectors.toList());
        pipelineInfo.setFinderPatterns(finderPatterns);
        pipelineInfo.setAlignmentPattern(alignmentPatterns.get(0));
        // Diagonal points are 0. and 1. 2. point is not involved in the diagonal. (A,B) -> On the diagonal, C not.
        List<Point> listOfPoints = getMostDistantFinderPoints(finderPatterns);
        Mat pointMatrix = Utils.convert2DPointsToMat(listOfPoints);
        System.out.println(pointMatrix.dump());
        Mat A = pointMatrix.row(0);
        Mat B = pointMatrix.row(1);
        Mat C = pointMatrix.row(2);
        // Get middle point of the QR Code.
        Mat middlePoint = getMiddlePoint(A, B);
        // Diagonal length
        double diagonalLength = Utils.getDistanceBetweenPoints(listOfPoints.get(0), listOfPoints.get(1));
        // M is the middle point, get the (C-M) vector.
        Mat c_m_vector = new Mat();
        Core.subtract(C, middlePoint, c_m_vector);
        System.out.println(c_m_vector.dump());
        Core.normalize(c_m_vector, c_m_vector);
        System.out.println(c_m_vector.dump());
        System.out.println(pointMatrix.dump());
        System.out.println(middlePoint.dump());
        System.out.println(c_m_vector.dump());
        // Obtain four reference vector by rotating diffVec by 45, 135, 225, 315 degrees.
        Mat[] referenceVectors = getReferenceVectors(c_m_vector);
        // Find the up vector, the vector points to the upper edge of the QR Code.
        try {
            findUpVector(c_m_vector, referenceVectors, middlePoint, qrCodePointList, pipelineInfo);
        } catch (Exception e) {
            e.printStackTrace();
        }
        double upVectorLength = Math.cos(45.0 / (180.0 / Math.PI)) * (diagonalLength / 2.0);
        Mat upVectorScaled = new Mat();
        Core.multiply(pipelineInfo.getRotationUpVector(), new Scalar(upVectorLength), upVectorScaled);
        Mat displacedMiddlePoint = new Mat();
        Core.add(middlePoint, upVectorScaled, displacedMiddlePoint);
        Mat upVectorLine = new Mat();
        Core.vconcat(new ArrayList<Mat>(Arrays.asList(middlePoint, displacedMiddlePoint)), upVectorLine);
        pipelineInfo.setRotationUpVectorStartFinishPoints(upVectorLine);
        return pipelineInfo;
    }
}
