package bandrol_training.model.QRCodeReading;

import org.opencv.core.Point;

public class QRCodePoint {
    private Point location;
    private QRCodePointTypes pointType;

    public QRCodePoint(Point loc, QRCodePointTypes type)
    {
        location = loc;
        pointType = type;
    }

    public Point getLocation() {
        return location;
    }

    public QRCodePointTypes getPointType() {
        return pointType;
    }
}
