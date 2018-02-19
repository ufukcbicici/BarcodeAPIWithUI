package sample.model.QRCodeReading;

import org.opencv.core.Mat;
import org.opencv.core.Point;

import java.awt.image.BufferedImage;
import java.util.List;

public class PipelineInfo {
    private double angle;
    private Point rotationCenter;
    private Mat rotationUpVector;
    private Mat rotationUpVectorStartFinishPoints;
    private BufferedImage originalImage;
    private BufferedImage rotatedImage;
    private List<QRCodePoint> finderPatterns;
    private QRCodePoint alignmentPattern;

    public double getAngle() {
        return angle;
    }

    public void setAngle(double angle) {
        this.angle = angle;
    }

    public Point getRotationCenter() {
        return rotationCenter;
    }

    public void setRotationCenter(Point rotationCenter) {
        this.rotationCenter = rotationCenter;
    }

    public Mat getRotationUpVector() {
        return rotationUpVector;
    }

    public void setRotationUpVector(Mat rotationUpVector) {
        this.rotationUpVector = rotationUpVector;
    }

    public Mat getRotationUpVectorStartFinishPoints() {
        return rotationUpVectorStartFinishPoints;
    }

    public void setRotationUpVectorStartFinishPoints(Mat rotationUpVectorStartFinishPoints) {
        this.rotationUpVectorStartFinishPoints = rotationUpVectorStartFinishPoints;
    }

    public BufferedImage getOriginalImage() {
        return originalImage;
    }

    public void setOriginalImage(BufferedImage originalImage) {
        this.originalImage = originalImage;
    }

    public BufferedImage getRotatedImage() {
        return rotatedImage;
    }

    public void setRotatedImage(BufferedImage rotatedImage) {
        this.rotatedImage = rotatedImage;
    }

    public List<QRCodePoint> getFinderPatterns() {
        return finderPatterns;
    }

    public void setFinderPatterns(List<QRCodePoint> finderPatterns) {
        this.finderPatterns = finderPatterns;
    }

    public QRCodePoint getAlignmentPattern() {
        return alignmentPattern;
    }

    public void setAlignmentPattern(QRCodePoint alignmentPattern) {
        this.alignmentPattern = alignmentPattern;
    }
}
