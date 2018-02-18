package sample.model.QRCodeReading;

import java.awt.image.BufferedImage;
import org.opencv.core.Point;

import java.util.List;


public interface IQRCodeReader {
    public List<QRCodePoint> detectQRCode(BufferedImage img, List<Point> cornerPoints);
}
