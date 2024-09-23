public class SerializingTranscoderTest extends BaseTranscoderCase { private SerializingTranscoder tc ; private TranscoderUtils tu ; @ Override protected void setUp ( ) throws Exception { super . setUp ( ) ; tc = new SerializingTranscoder ( ) ; setTranscoder ( tc ) ; tu = new TranscoderUtils ( true ) ; } public void testNonserializable ( ) throws Exception { try { tc . encode ( new Object ( ) ) ; fail ( "Processed a non-serializable object . " ) ; } catch ( IllegalArgumentException e ) { } } public void testJsonObject ( ) { String json = " { \"aaaaaaaaaaaaaaaaaaaaaaaaa\" : " + "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" } " ; tc . setCompressionThreshold ( 8 ) ; CachedData cd = tc . encode ( json ) ; assertFalse ( "Flags shows JSON was compressed" , ( cd . getFlags ( ) & ( 1L < < SerializingTranscoder . COMPRESSED ) ) != 0 ) ; assertTrue ( "JSON was incorrectly encoded" , Arrays . equals ( json . getBytes ( ) , cd . getData ( ) ) ) ; assertEquals ( "JSON was harmed , should not have been" , json , tc . decode ( cd ) ) ; } public void testCompressedStringNotSmaller ( ) throws Exception { String s1 = "This is a test simple string that will not be compressed . " ; tc . setCompressionThreshold ( 8 ) ; CachedData cd = tc . encode ( s1 ) ; assertEquals ( 0 , cd . getFlags ( ) ) ; assertTrue ( Arrays . equals ( s1 . getBytes ( ) , cd . getData ( ) ) ) ; assertEquals ( s1 , tc . decode ( cd ) ) ; } public void testCompressedString ( ) throws Exception { String s1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" ; tc . setCompressionThreshold ( 8 ) ; CachedData cd = tc . encode ( s1 ) ; assertEquals ( SerializingTranscoder . COMPRESSED , cd . getFlags ( ) ) ; assertFalse ( Arrays . equals ( s1 . getBytes ( ) , cd . getData ( ) ) ) ; assertEquals ( s1 , tc . decode ( cd ) ) ; } public void testObject ( ) throws Exception { Calendar c = Calendar . getInstance ( ) ; CachedData cd = tc . encode ( c ) ; assertEquals ( SerializingTranscoder . SERIALIZED , cd . getFlags ( ) ) ; assertEquals ( c , tc . decode ( cd ) ) ; } public void testCompressedObject ( ) throws Exception { tc . setCompressionThreshold ( 8 ) ; Calendar c = Calendar . getInstance ( ) ; CachedData cd = tc . encode ( c ) ; assertEquals ( SerializingTranscoder . SERIALIZED | SerializingTranscoder . COMPRESSED , cd . getFlags ( ) ) ; assertEquals ( c , tc . decode ( cd ) ) ; } public void testUnencodeable ( ) throws Exception { try { CachedData cd = tc . encode ( new Object ( ) ) ; fail ( "Should fail to serialize , got" + cd ) ; } catch ( IllegalArgumentException e ) { } } public void testUndecodeable ( ) throws Exception { CachedData cd = new CachedData ( Integer . MAX_VALUE & ~ ( SerializingTranscoder . COMPRESSED | SerializingTranscoder . SERIALIZED ) , tu . encodeInt ( Integer . MAX_VALUE ) , tc . getMaxSize ( ) ) ; assertNull ( tc . decode ( cd ) ) ; } public void testUndecodeableSerialized ( ) throws Exception { CachedData cd = new CachedData ( SerializingTranscoder . SERIALIZED , tu . encodeInt ( Integer . MAX_VALUE ) , tc . getMaxSize ( ) ) ; assertNull ( tc . decode ( cd ) ) ; } public void testUndecodeableCompressed ( ) throws Exception { CachedData cd = new CachedData ( SerializingTranscoder . COMPRESSED , tu . encodeInt ( Integer . MAX_VALUE ) , tc . getMaxSize ( ) ) ; System . out . println ( "got " + tc . decode ( cd ) ) ; assertNull ( tc . decode ( cd ) ) ; } @ Override protected int getStringFlags ( ) { return 0 ; } }