package sample.model;

import org.opencv.core.Point;
import sample.model.OrientationFinding.OrientationFinder;
import sample.model.QRCodeReading.IQRCodeReader;
import sample.model.QRCodeReading.QRCodePoint;
import sample.model.QRCodeReading.QRCodeReaders.ZxingQRCodeReader;

import java.awt.image.BufferedImage;
import java.util.List;

public class Algorithm {
    public static IQRCodeReader qrCodeReader;

    public static void InitAlgorithm()
    {
        qrCodeReader = new ZxingQRCodeReader();
    }

    public static List<QRCodePoint> getQRCodePoints(BufferedImage img, List<Point> corners)
    {
        return qrCodeReader.detectQRCode(img, corners);
    }

    public static void getOrientationFromQRCodePoints(List<QRCodePoint> qrCodePointList)
    {
        OrientationFinder.findOrientationFromQRCode(qrCodePointList);
    }

    public static void execute(BufferedImage img, List<Point> corners)
    {
        // Step 1: Detect the QR Code with Zxing library
        List<QRCodePoint> qrCodePointList = getQRCodePoints(img, corners);
        if (qrCodePointList == null)
            return;
        // Step 2: Get the QR Code orientation.
        getOrientationFromQRCodePoints(qrCodePointList);
    }
}
