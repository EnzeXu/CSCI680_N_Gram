public class RandomGeohashes { public static Iterable < GeoHash > fullRange ( ) { return new Iterable < GeoHash > ( ) { @ Override public Iterator < GeoHash > iterator ( ) { Random rand = new Random ( ) ; List < GeoHash > hashes = new ArrayList < > ( ) ; for ( double lat = -90 ; lat < = 90 ; lat += rand . nextDouble ( ) + 1 . 45 ) { for ( double lon = -180 ; lon < = 180 ; lon += rand . nextDouble ( ) + 1 . 54 ) { for ( int precisionChars = 6 ; precisionChars < = 12 ; precisionChars++ ) { GeoHash gh = GeoHash . withCharacterPrecision ( lat , lon , precisionChars ) ; hashes . add ( gh ) ; } } } return hashes . iterator ( ) ; } } ; } private static final Random rand = new Random ( 9817298371L ) ; public static GeoHash create ( ) { return GeoHash . withBitPrecision ( randomLatitude ( ) , randomLongitude ( ) , randomPrecision ( ) ) ; } public static GeoHash createWith5BitsPrecision ( ) { return GeoHash . withCharacterPrecision ( randomLatitude ( ) , randomLongitude ( ) , randomCharacterPrecision ( ) ) ; } public static GeoHash createWithPrecision ( int precision ) { return GeoHash . withBitPrecision ( randomLatitude ( ) , randomLongitude ( ) , precision ) ; } private static double randomLatitude ( ) { return ( rand . nextDouble ( ) - 0 . 5 ) * 180 ; } private static double randomLongitude ( ) { return ( rand . nextDouble ( ) - 0 . 5 ) * 360 ; } private static int randomPrecision ( ) { return rand . nextInt ( 60 ) + 5 ; } private static int randomCharacterPrecision ( ) { return rand . nextInt ( 12 ) + 1 ; } }