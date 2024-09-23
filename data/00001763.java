public class BigDecimalSerialization extends Configured implements Serialization < BigDecimal > { public static class BigDecimalDeserializer implements Deserializer < BigDecimal > { private DataInputStream in ; @ Override public void open ( InputStream in ) throws IOException { if ( in instanceof DataInputStream ) this . in = ( DataInputStream ) in ; else this . in = new DataInputStream ( in ) ; } @ Override public BigDecimal deserialize ( BigDecimal existing ) throws IOException { int len = in . readInt ( ) ; byte [ ] valueBytes = new byte [ len ] ; in . readFully ( valueBytes ) ; BigInteger value = new BigInteger ( valueBytes ) ; return new BigDecimal ( value , in . readInt ( ) ) ; } @ Override public void close ( ) throws IOException { in . close ( ) ; } } public static class BigDecimalSerializer implements Serializer < BigDecimal > { private DataOutputStream out ; @ Override public void open ( OutputStream out ) throws IOException { if ( out instanceof DataOutputStream ) this . out = ( DataOutputStream ) out ; else this . out = new DataOutputStream ( out ) ; } @ Override public void serialize ( BigDecimal bigDecimal ) throws IOException { BigInteger value = bigDecimal . unscaledValue ( ) ; byte [ ] valueBytes = value . toByteArray ( ) ; out . writeInt ( valueBytes . length ) ; out . write ( valueBytes ) ; out . writeInt ( bigDecimal . scale ( ) ) ; } @ Override public void close ( ) throws IOException { out . close ( ) ; } } public BigDecimalSerialization ( ) { } @ Override public boolean accept ( Class < ? > c ) { return BigDecimal . class == c ; } @ Override public Serializer < BigDecimal > getSerializer ( Class < BigDecimal > c ) { return new BigDecimalSerializer ( ) ; } @ Override public Deserializer < BigDecimal > getDeserializer ( Class < BigDecimal > c ) { return new BigDecimalDeserializer ( ) ; } }