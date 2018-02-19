package sample.model.QRCodeReading.QRCodeReaders;

import com.google.zxing.qrcode.detector.FinderPattern;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import sample.model.QRCodeReading.IQRCodeReader;
import com.google.zxing.*;
import com.google.zxing.client.j2se.BufferedImageLuminanceSource;
import com.google.zxing.common.HybridBinarizer;
import sample.model.QRCodeReading.QRCodePoint;
import sample.model.QRCodeReading.QRCodePointTypes;
import sample.model.Utils;

import java.awt.image.BufferedImage;
import java.util.*;

public class ZxingQRCodeReader implements IQRCodeReader{
    @Override
    public List<QRCodePoint> detectQRCode(BufferedImage img, List<Point> cornerPoints) {
        //Focus on the region
        Rect2d boundingBox = Utils.getTightestBoundingRectangle(cornerPoints);
        BufferedImage focusImg = Utils.cropImage(img, boundingBox);
        //Utils.showImageInPopup(focusImg);
        // Image focusImg = img;
        //Extract QR Code info using ZXing
        LuminanceSource source = new BufferedImageLuminanceSource(focusImg);
        BinaryBitmap bitmap = new BinaryBitmap(new HybridBinarizer(source));
        Map<DecodeHintType,Object> hintsMap = new EnumMap<>(DecodeHintType.class);
        hintsMap.put(DecodeHintType.TRY_HARDER, Boolean.TRUE);
        hintsMap.put(DecodeHintType.POSSIBLE_FORMATS, EnumSet.of(BarcodeFormat.QR_CODE));
        List<QRCodePoint> pointList = new ArrayList<>();
        try {
            Result result = new MultiFormatReader().decode(bitmap, hintsMap);
            System.out.println("QR Code found.");
            System.out.println(result);
            for(int i=0;i<result.getResultPoints().length;i++)
            {
                // Convert the coordinates to original frame
                Point opencvPoint = new Point(boundingBox.x + result.getResultPoints()[i].getX(),
                        boundingBox.y + result.getResultPoints()[i].getY());
                if(result.getResultPoints()[i] instanceof FinderPattern)
                {
                    pointList.add(new QRCodePoint(opencvPoint, QRCodePointTypes.FINDER_PATTERN));
                }
                else
                {
                    pointList.add(new QRCodePoint(opencvPoint, QRCodePointTypes.ALIGNMENT_PATTERN));
                }
            }
        } catch (NotFoundException e) {
            System.out.println("There is no QR code in the image");
            return null;
        }
        return pointList;
    }
}
