package sample.model;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {
    public static Rect2d getTightestBoundingRectangle(List<Point> points)
    {
        double minx = Double.MAX_VALUE;
        double maxx = Double.MIN_VALUE;
        double miny = Double.MAX_VALUE;
        double maxy = Double.MIN_VALUE;
        for (Point point : points)
        {
            if (point.x < minx)
                minx = point.x;
            if(point.x > maxx)
                maxx = point.x;
            if(point.y < miny)
                miny = point.y;
            if(point.y > maxy)
                maxy = point.y;
        }
        return new Rect2d(minx, miny, maxx - minx, maxy - miny);
    }

    public static BufferedImage cropImage(BufferedImage img, Rect2d cropArea)
    {
        return img.getSubimage((int)cropArea.x, (int)cropArea.y, (int)cropArea.width, (int)cropArea.height);
    }

    public static void showImageInPopup(BufferedImage img)
    {
        JLabel lbl = new JLabel(new ImageIcon(img));
        JOptionPane.showMessageDialog(null, lbl, "ImageDialog",
                JOptionPane.PLAIN_MESSAGE, null);

    }

    public static double getDistanceBetweenPoints(Point p0, Point p1)
    {
        double dist_x = p1.x - p0.x;
        double dist_y = p1.y - p0.y;
        return Math.sqrt(dist_x*dist_x + dist_y*dist_y);
    }

    public static double [] solveQuadraticEquation(double A, double B, double C)
    {
        double determinant = B * B - 4 * A * C;
        double root1, root2;
        if(determinant > 0)
        {
            root1 = (-B + Math.sqrt(determinant)) / (2 * A);
            root2 = (-B - Math.sqrt(determinant)) / (2 * A);
            return new double[]{root1, root2};
        }
        else if(determinant == 0)
        {
            root1 = root2 = -B / (2 * A);
            return new double[]{root1};
        }
        else
        {
            double realPart = -B / (2 * A);
            double imaginaryPart = Math.sqrt(-determinant) / (2 * A);
            return new double[]{};
        }
    }

    public static Mat getVectorsWithAngleToRefVector2D(Mat refVector, double angle) throws Exception
    {
        // For numerical stability
        int pivotIndex = -1;
        int nonPivotIndex = -1;
        double cx, cy = 0.0;
        if (Math.abs(refVector.get(0,0)[0]) > Math.abs(refVector.get(0,0)[1]))
        {
            pivotIndex = 0;
            cx = refVector.get(0,0)[0];
            cy = refVector.get(0,0)[1];
        }
        else
        {
            pivotIndex = 1;
            cx = refVector.get(0,0)[1];
            cy = refVector.get(0,0)[0];
        }
        double cosTerm = Math.cos( (angle * Math.PI)/180.0 );
        double A = (cy * cy)/(cx * cx) + 1.0;
        double B = - (2.0 * cy * cosTerm)/(cx * cx);
        double C = (cosTerm * cosTerm)/(cx * cx) - 1.0;
        double [] roots = solveQuadraticEquation(A, B, C);
        double v_y0 = Double.MAX_VALUE;
        double v_x0 = Double.MAX_VALUE;
        double v_y1 = Double.MAX_VALUE;
        double v_x1 = Double.MAX_VALUE;
        if (pivotIndex == 0)
        {
            v_y0 = roots[0];
            v_x0 = Math.sqrt(1.0 - v_y0*v_y0);
            v_y1 = roots[1];
            v_x1 = Math.sqrt(1.0 - v_y1*v_y1);
        }
        else
        {
            v_x0 = roots[0];
            v_y0 = Math.sqrt(1.0 - v_x0*v_x0);
            v_x1 = roots[1];
            v_y1 = Math.sqrt(1.0 - v_x1*v_x1);
        }
        Point vec0 = new Point(v_x0, v_y0);
        Point vec1 = new Point(v_x1, v_y1);
        return Converters.vector_Point_to_Mat(new ArrayList<>(Arrays.asList(vec0, vec1)));
    }

    public static Mat convert2DPointsToMat(List<Point> points)
    {
        int count = (points != null) ? points.size() : 0;
        Mat res = new Mat(count, 2, CvType.CV_64F);
        for (int i = 0; i < count; i++)
        {
            Point p = points.get(i);
            res.put(i, 0, p.x);
            res.put(i, 1, p.y);
        }
        return res;
    }

    public static List<Point> convertMatTo2DPoints(Mat mat) throws Exception {
        if(mat.cols() != 2)
        {
            throw new Exception("Mat does not have two columns!");
        }
        List<Point> pointList = new ArrayList<>();
        for(int i=0;i<mat.rows();i++)
        {
            Point p = new Point(mat.get(i,0)[0], mat.get(i,1)[0]);
            pointList.add(p);
        }
        return pointList;
    }

    public static Mat rotate2DPoint(Mat point, Point center, double angle) throws Exception {
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        Mat pointToRotate = point.clone();
        boolean doTranspose = false;
        if(point.rows() == 1 && point.cols() == 2)
        {
            Core.transpose(point, pointToRotate);
            doTranspose = true;
        }
        else if(point.rows() != 2 || point.cols() != 1)
            throw new Exception("Unsupported point dimensions.");
        Mat rotated = new Mat();
        Mat delta = new Mat();
        Mat point3D = new Mat(3, 1, CvType.CV_64F);
        point3D.put(0,0,pointToRotate.get(0,0)[0]);
        point3D.put(1,0,pointToRotate.get(1,0)[0]);
        point3D.put(2,0,1.0);
        Core.gemm(rotationMatrix, point3D,1.0, delta, 0.0, rotated);
        // System.out.println(rotated.dump());
        if(doTranspose)
            Core.transpose(rotated, rotated);
        return rotated;
    }

//    static BufferedImage Mat2BufferedImage(Mat matrix)throws Exception {
//        MatOfByte mob=new MatOfByte();
//        Imgcodecs.imencode(".jpg", matrix, mob);
//        byte ba[]=mob.toArray();
//
//        BufferedImage bi=ImageIO.read(new ByteArrayInputStream(ba));
//        return bi;
//    }
}
