public class GeoHashCircleQuery implements GeoHashQuery, Serializable { private static final long serialVersionUID = 1263295371663796291L; private double radius; private GeoHashBoundingBoxQuery query; private WGS84Point center; public GeoHashCircleQuery(WGS84Point center, double radius) { this.radius = radius; this.center = center; WGS84Point northEastCorner = VincentyGeodesy.moveInDirection(VincentyGeodesy.moveInDirection(center, 0, radius), 90, radius); WGS84Point southWestCorner = VincentyGeodesy.moveInDirection(VincentyGeodesy.moveInDirection(center, 180, radius), 270, radius); BoundingBox bbox = new BoundingBox(southWestCorner, northEastCorner); query = new GeoHashBoundingBoxQuery(bbox); } @Override public boolean contains(GeoHash hash) { return query.contains(hash); } @Override public String getWktBox() { return query.getWktBox(); } @Override public List<GeoHash> getSearchHashes() { return query.getSearchHashes(); } @Override public String toString() { return "Cicle Query [center=" + center + ", radius=" + getRadiusString() + "]"; } private String getRadiusString() { if (radius > 1000) { return radius / 1000 + "km"; } else { return radius + "m"; } } @Override public boolean contains(WGS84Point point) { return query.contains(point); } }