public class Glyph { protected short leftSideBearing; protected int advanceWidth; private Point[] points; public Glyph(GlyphDescription gd, short lsb, int advance) { leftSideBearing = lsb; advanceWidth = advance; describe(gd); } public int getAdvanceWidth() { return advanceWidth; } public short getLeftSideBearing() { return leftSideBearing; } public Point getPoint(int i) { return points[i]; } public int getPointCount() { return points.length; } public void reset() { } public void scale(float factor) { if(points!=null){ for (int i = 0; i < points.length; i++) { points[i].x *= factor; points[i].y *= -factor; } } leftSideBearing = (short)((float)leftSideBearing * factor); advanceWidth = (int)((float)advanceWidth * factor); } private void describe(GlyphDescription gd) { int endPtIndex = 0; points = new Point[gd.getPointCount() + 2]; for (int i = 0; i < gd.getPointCount(); i++) { boolean endPt = gd.getEndPtOfContours(endPtIndex) == i; if (endPt) { endPtIndex++; } points[i] = new Point( gd.getXCoordinate(i), gd.getYCoordinate(i), (gd.getFlags(i) & GlyfDescript.onCurve) != 0, endPt); } points[gd.getPointCount()] = new Point(0, 0, true, true); points[gd.getPointCount()+1] = new Point(advanceWidth, 0, true, true); } }