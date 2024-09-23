public class SortedValuesPlatformTest extends PlatformTestCase { private String apacheCommonRegex = TestConstants.APACHE_COMMON_REGEX; private RegexParser apacheCommonParser = new RegexParser( new Fields( "ip", "time", "method", "event", "status", "size" ), apacheCommonRegex, new int[]{ 1, 2, 3, 4, 5, 6} ); public SortedValuesPlatformTest() { super( false ); } @Test public void testCoGroupComparatorValues() throws Exception { runCoGroupComparatorTest( "cogroupcompareforward", false ); } @Test public void testCoGroupComparatorValuesReversed() throws Exception { runCoGroupComparatorTest( "cogroupcomparereversed", true ); } private void runCoGroupComparatorTest( String path, boolean reverseSort ) throws IOException, ParseException { getPlatform().copyFromLocal( inputFileApache200 ); getPlatform().copyFromLocal( inputFileIps ); Tap sourceApache = getPlatform().getTextFile( inputFileApache200 ); Tap sourceIP = getPlatform().getTextFile( inputFileIps ); Tap sink = getPlatform().getTextFile( getOutputPath( path ), SinkMode.REPLACE ); Pipe apachePipe = new Pipe( "apache" ); apachePipe = new Each( apachePipe, new Fields( "line" ), apacheCommonParser ); apachePipe = new Each( apachePipe, new Insert( new Fields( "col" ), 1 ), Fields.ALL ); apachePipe = new Each( apachePipe, new Fields( "ip" ), new RegexParser( new Fields( "octet" ), "^[^.]*" ), new Fields( "col", "status", "event", "octet", "size" ) ); apachePipe = new Each( apachePipe, new Fields( "octet" ), new Identity( long.class ), Fields.REPLACE ); Fields groupApache = new Fields( "octet" ); groupApache.setComparator( "octet", getPlatform().getLongComparator( reverseSort ) ); Pipe ipPipe = new Pipe( "ip" ); ipPipe = new Each( ipPipe, new Fields( "line" ), new Identity( new Fields( "rawip" ) ) ); ipPipe = new Each( ipPipe, new Fields( "rawip" ), new RegexParser( new Fields( "rawoctet" ), "^[^.]*" ), new Fields( "rawoctet" ) ); ipPipe = new Each( ipPipe, new Fields( "rawoctet" ), new Identity( long.class ), Fields.REPLACE ); Fields groupIP = new Fields( "rawoctet" ); groupIP.setComparator( "rawoctet", getPlatform().getLongComparator( reverseSort ) ); Pipe pipe = new CoGroup( apachePipe, groupApache, ipPipe, groupIP ); pipe = new Each( pipe, new Identity() ); Map<Object, Object> properties = getProperties(); if( getPlatform().isMapReduce() && getPlatform().getNumMapTasks( properties ) != null ) getPlatform().setNumMapTasks( properties, 13 ); Map sources = new HashMap(); sources.put( "apache", sourceApache ); sources.put( "ip", sourceIP ); Flow flow = getPlatform().getFlowConnector().connect( sources, sink, pipe ); flow.complete(); validateFile( sink, 199, 16, reverseSort, 5 ); } @Test public void testComprehensiveGroupBy() throws IOException { Boolean[][] testArray = new Boolean[][]{ {false, null, false}, {true, null, false}, {false, null, true}, {true, null, true}, {false, false, false}, {true, false, false}, {true, true, false}, {false, true, false}, {false, false, true}, {true, false, true}, {true, true, true}, {false, true, true} }; for( int i = 0; i < testArray.length; i++ ) runComprehensiveCase( testArray[ i ], false ); for( int i = 0; i < testArray.length; i++ ) runComprehensiveCase( testArray[ i ], true ); } private void runComprehensiveCase( Boolean[] testCase, boolean useCollectionsComparator ) throws IOException { getPlatform().copyFromLocal( inputFileCrossNulls ); String test = Util.join( testCase, "_", true ) + "_" + useCollectionsComparator; String path = "comprehensive/" + test; Tap source = getPlatform().getTextFile( new Fields( "line" ), inputFileCrossNulls ); Tap sink = getPlatform().getDelimitedFile( new Fields( "num", "lower", "upper" ).applyTypes( Long.class, String.class, String.class ), " ", getOutputPath( path ), SinkMode.REPLACE ); sink.getScheme().setNumSinkParts( 1 ); Pipe pipe = new Pipe( "comprehensivesort" ); pipe = new Each( pipe, new Fields( "line" ), new RegexSplitter( new Fields( "num", "lower", "upper" ), "\\s" ) ); pipe = new Each( pipe, new Fields( "num" ), new Identity( Long.class ), Fields.REPLACE ); Fields groupFields = new Fields( "num" ); if( testCase[ 0 ] ) groupFields.setComparator( "num", useCollectionsComparator ? new NullSafeReverseComparator() : getPlatform().getLongComparator( true ) ); Fields sortFields = null; if( testCase[ 1 ] != null ) { sortFields = new Fields( "upper" ); if( testCase[ 1 ] ) sortFields.setComparator( "upper", useCollectionsComparator ? new NullSafeReverseComparator() : getPlatform().getStringComparator( true ) ); } pipe = new GroupBy( pipe, groupFields, sortFields, testCase[ 2 ] ); Map<Object, Object> properties = getProperties(); if( getPlatform().isMapReduce() && getPlatform().getNumMapTasks( properties ) != null ) getPlatform().setNumMapTasks( properties, 13 ); Flow flow = getPlatform().getFlowConnector().connect( source, sink, pipe ); flow.complete(); validateCase( test, testCase, sink ); } private void validateCase( String test, Boolean[] testCase, Tap sink ) throws IOException { TupleEntryIterator iterator = sink.openForRead( getPlatform().getFlowProcess() ); LinkedHashMap<Long, List<String>> group = new LinkedHashMap<Long, List<String>>(); while( iterator.hasNext() ) { Tuple tuple = iterator.next().getTuple(); if( !group.containsKey( tuple.getLong( 0 ) ) ) group.put( tuple.getLong( 0 ), new ArrayList<String>() ); group.get( tuple.getLong( 0 ) ).add( tuple.getString( 2 ) ); } boolean groupIsReversed = testCase[ 0 ]; if( testCase[ 2 ] ) groupIsReversed = !groupIsReversed; compare( "grouping+" + test, groupIsReversed, group.keySet() ); if( testCase[ 1 ] == null ) return; boolean valueIsReversed = testCase[ 1 ]; if( testCase[ 2 ] ) valueIsReversed = !valueIsReversed; for( Long grouping : group.keySet() ) compare( "values+" + test, valueIsReversed, group.get( grouping ) ); } private void compare( String test, boolean isReversed, Collection values ) { List<Object> groups = new ArrayList<Object>( values ); List<Object> sortedGroups = new ArrayList<Object>( groups ); Collections.sort( sortedGroups, isReversed ? Collections.reverseOrder() : null ); assertEquals( test, sortedGroups, groups ); } @Test public void testSortFails() throws Exception { Tap source = getPlatform().getTextFile( "foosource" ); Tap sink = getPlatform().getTextFile( "foosink" ); Pipe pipe = new Pipe( "apache" ); pipe = new Each( pipe, new Fields( "line" ), apacheCommonParser ); pipe = new Each( pipe, new Insert( new Fields( "col" ), 1 ), Fields.ALL ); pipe = new Each( pipe, new Fields( "time" ), new DateParser( "dd/MMM/yyyy:HH:mm:ss Z" ), new Fields( "col", "status", "ts", "event", "ip", "size" ) ); pipe = new GroupBy( pipe, new Fields( "col" ), new Fields( "does-not-exist" ) ); pipe = new Each( pipe, new Identity() ); try { getPlatform().getFlowConnector().connect( source, sink, pipe ); fail( "did not throw exception" ); } catch( Exception exception ) { } } private void validateFile( Tap tap, int length, int uniqueValues, boolean isReversed, int comparePosition ) throws IOException, ParseException { TupleEntryIterator iterator = tap.openForRead( getPlatform().getFlowProcess() ); Set<Long> values = new HashSet<Long>(); long lastValue = isReversed ? Long.MAX_VALUE : Long.MIN_VALUE; int count = 0; while( iterator.hasNext() ) { Tuple tuple = iterator.next().getTuple(); count++; tuple = new Tuple( (Object[]) tuple.getString( 1 ).split( "\t" ) ); long value = tuple.getLong( comparePosition ); values.add( value ); if( isReversed ) assertTrue( "out of order in " + tap, lastValue >= value ); else assertTrue( "out of order in " + tap, lastValue <= value ); lastValue = value; } if( length != -1 ) assertEquals( "length of " + tap, length, count ); if( uniqueValues != -1 ) assertEquals( "unique values of " + tap, uniqueValues, values.size() ); } }