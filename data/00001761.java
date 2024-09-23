public class BytesSerialization extends Configured implements Comparison < byte [ ] > , Serialization < byte [ ] > { public static class RawBytesDeserializer implements Deserializer < byte [ ] > { private DataInputStream in ; @ Override public void open ( InputStream in ) throws IOException { if ( in instanceof DataInputStream ) this . in = ( DataInputStream ) in ; else this . in = new DataInputStream ( in ) ; } @ Override public byte [ ] deserialize ( byte [ ] existing ) throws IOException { int len = in . readInt ( ) ; byte [ ] bytes = existing != null && existing . length == len ? existing : new byte [ len ] ; in . readFully ( bytes ) ; return bytes ; } @ Override public void close ( ) throws IOException { in . close ( ) ; } } public static class RawBytesSerializer implements Serializer < byte [ ] > { private DataOutputStream out ; @ Override public void open ( OutputStream out ) throws IOException { if ( out instanceof DataOutputStream ) this . out = ( DataOutputStream ) out ; else this . out = new DataOutputStream ( out ) ; } @ Override public void serialize ( byte [ ] bytes ) throws IOException { out . writeInt ( bytes . length ) ; out . write ( bytes ) ; } @ Override public void close ( ) throws IOException { out . close ( ) ; } } public BytesSerialization ( ) { } @ Override public boolean accept ( Class < ? > c ) { return byte [ ] . class == c ; } @ Override public Serializer < byte [ ] > getSerializer ( Class < byte [ ] > c ) { return new RawBytesSerializer ( ) ; } @ Override public Deserializer < byte [ ] > getDeserializer ( Class < byte [ ] > c ) { return new RawBytesDeserializer ( ) ; } @ Override public Comparator < byte [ ] > getComparator ( Class < byte [ ] > type ) { return new BytesComparator ( ) ; } }