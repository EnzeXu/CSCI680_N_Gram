public class TupleTest extends CascadingTestCase { public TupleTest() { } @Test public void testWritableCompareReadWrite() throws IOException { Tuple aTuple = new Tuple( new TestWritableComparable( "Just My Luck" ), "ClaudiaPuig", "3.0", "LisaRose", "3.0", true ); ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(); TupleOutputStream dataOutputStream = new HadoopTupleOutputStream( byteArrayOutputStream, new TupleSerialization().getElementWriter() ); dataOutputStream.writeTuple( aTuple ); dataOutputStream.flush(); ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream( byteArrayOutputStream.toByteArray() ); TupleInputStream dataInputStream = new HadoopTupleInputStream( byteArrayInputStream, new TupleSerialization().getElementReader() ); Tuple newTuple = new Tuple(); dataInputStream.readTuple( newTuple ); assertEquals( aTuple, newTuple ); } @Test public void testWritableCompare() { Tuple aTuple = new Tuple( new TestWritableComparable( "Just My Luck" ), "ClaudiaPuig", "3.0", "LisaRose", "3.0" ); Tuple bTuple = new Tuple( new TestWritableComparable( "Just My Luck" ), "ClaudiaPuig", "3.0", "LisaRose", "3.0" ); assertEquals( "not equal: aTuple", bTuple, aTuple ); assertTrue( "not equal than: aTuple = bTuple", aTuple.compareTo( bTuple ) == 0 ); bTuple = new Tuple( new TestWritableComparable( "Just My Luck" ), "ClaudiaPuig", "3.0", "LisaRose", "2.0" ); assertTrue( "not less than: aTuple < bTuple", aTuple.compareTo( bTuple ) > 0 ); } }