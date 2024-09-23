public class SpillableTupleHadoopTest extends CascadingTestCase { public SpillableTupleHadoopTest() { super(); } @Test public void testSpillList() { long time = System.currentTimeMillis(); performListTest( 5, 50, null, 0 ); performListTest( 49, 50, null, 0 ); performListTest( 50, 50, null, 0 ); performListTest( 51, 50, null, 1 ); performListTest( 499, 50, null, 9 ); performListTest( 500, 50, null, 9 ); performListTest( 501, 50, null, 10 ); System.out.println( "time = " + ( System.currentTimeMillis() - time ) ); } @Test public void testSpillListCompressed() { GzipCodec codec = ReflectionUtils.newInstance( GzipCodec.class, new Configuration() ); long time = System.currentTimeMillis(); performListTest( 5, 50, codec, 0 ); performListTest( 49, 50, codec, 0 ); performListTest( 50, 50, codec, 0 ); performListTest( 51, 50, codec, 1 ); performListTest( 499, 50, codec, 9 ); performListTest( 500, 50, codec, 9 ); performListTest( 501, 50, codec, 10 ); System.out.println( "time = " + ( System.currentTimeMillis() - time ) ); } private void performListTest( int size, int threshold, CompressionCodec codec, int spills ) { Configuration jobConf = new Configuration(); jobConf.set( "io.serializations", TestSerialization.class.getName() + "," + WritableSerialization.class.getName() ); jobConf.set( "cascading.serialization.tokens", "1000=" + BooleanWritable.class.getName() + ",10001=" + Text.class.getName() ); HadoopSpillableTupleList list = new HadoopSpillableTupleList( threshold, codec, jobConf ); for( int i = 0; i < size; i++ ) { String aString = "string number " + i; double random = Math.random(); list.add( new Tuple( i, aString, random, new Text( aString ), new TestText( aString ), new Tuple( "inner tuple", new BytesWritable( aString.getBytes() ) ) ) ); } assertEquals( "not equal: list.size();", size, list.size() ); assertEquals( "not equal: list.getNumFiles()", spills, list.spillCount() ); int i = -1; int count = 0; for( Tuple tuple : list ) { int value = tuple.getInteger( 0 ); assertTrue( "wrong diff", value - i == 1 ); assertEquals( "wrong value", "string number " + count, tuple.getObject( 3 ).toString() ); assertEquals( "wrong value", "string number " + count, tuple.getObject( 4 ).toString() ); assertTrue( "wrong type", tuple.getObject( 5 ) instanceof Tuple ); BytesWritable bytesWritable = (BytesWritable) ( (Tuple) tuple.getObject( 5 ) ).getObject( 1 ); byte[] bytes = bytesWritable.getBytes(); String actual = new String( bytes, 0, bytesWritable.getLength() ); assertEquals( "wrong value", "string number " + count, actual ); i = value; count++; } assertEquals( "not equal: list.size();", size, count ); Iterator<Tuple> iterator = list.iterator(); assertEquals( "not equal: iterator.next().get(1)", "string number 0", iterator.next().getObject( 1 ) ); assertEquals( "not equal: iterator.next().get(1)", "string number 1", iterator.next().getObject( 1 ) ); } @Test public void testSpillMap() { long time = System.currentTimeMillis(); Configuration jobConf = new Configuration(); performMapTest( 5, 5, 100, 20, jobConf ); performMapTest( 5, 50, 100, 20, jobConf ); performMapTest( 50, 5, 200, 20, jobConf ); performMapTest( 500, 50, 7000, 20, jobConf ); System.out.println( "time = " + ( System.currentTimeMillis() - time ) ); } @Test public void testSpillMapCompressed() { long time = System.currentTimeMillis(); Configuration jobConf = new Configuration(); jobConf.set( SpillableProps.SPILL_CODECS, "org.apache.hadoop.io.compress.GzipCodec" ); performMapTest( 5, 5, 100, 20, jobConf ); performMapTest( 5, 50, 100, 20, jobConf ); performMapTest( 50, 5, 200, 20, jobConf ); performMapTest( 500, 50, 7000, 20, jobConf ); System.out.println( "time = " + ( System.currentTimeMillis() - time ) ); } private void performMapTest( int numKeys, int listSize, int mapThreshold, int listThreshold, Configuration jobConf ) { jobConf.set( "io.serializations", TestSerialization.class.getName() + "," + WritableSerialization.class.getName() ); jobConf.set( "cascading.serialization.tokens", "1000=" + BooleanWritable.class.getName() + ",10001=" + Text.class.getName() ); HadoopFlowProcess flowProcess = new HadoopFlowProcess( jobConf ); HadoopSpillableTupleMap map = new HadoopSpillableTupleMap( SpillableProps.defaultMapInitialCapacity, SpillableProps.defaultMapLoadFactor, mapThreshold, listThreshold, flowProcess ); Set<Integer> keySet = new HashSet<Integer>(); Random gen = new Random( 1 ); for( int i = 0; i < listSize * numKeys; i++ ) { String aString = "string number " + i; double random = Math.random(); double keys = numKeys / 3.0; int key = (int) ( gen.nextDouble() * keys + gen.nextDouble() * keys + gen.nextDouble() * keys ); Tuple tuple = new Tuple( i, aString, random, new Text( aString ), new TestText( aString ), new Tuple( "inner tuple", new BytesWritable( aString.getBytes() ) ) ); map.get( new Tuple( key ) ).add( tuple ); keySet.add( key ); } assertEquals( "not equal: map.size();", keySet.size(), map.size() ); } }