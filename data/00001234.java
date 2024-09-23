public class LimitFilterTest extends CascadingTestCase { private ConcreteCall operationCall; public LimitFilterTest() { } @Override public void setUp() throws Exception { super.setUp(); operationCall = new ConcreteCall(); } private TupleEntry getEntry( Tuple tuple ) { return new TupleEntry( Fields.size( tuple.size() ), tuple ); } private class TestFlowProcess extends FlowProcess<Object> { private int numTasks; private int taskNum; public TestFlowProcess( int numTasks, int taskNum ) { super( new FlowSession() ); this.numTasks = numTasks; this.taskNum = taskNum; } @Override public FlowProcess copyWith( Object object ) { return null; } @Override public int getNumProcessSlices() { return numTasks; } @Override public int getCurrentSliceNum() { return taskNum; } @Override public Object getProperty( String key ) { return null; } @Override public Collection<String> getPropertyKeys() { return null; } @Override public Object newInstance( String className ) { return null; } @Override public void keepAlive() { } @Override public void increment( Enum counter, long amount ) { } @Override public void increment( String group, String counter, long amount ) { } @Override public long getCounterValue( Enum counter ) { return 0; } @Override public long getCounterValue( String group, String counter ) { return 0; } @Override public void setStatus( String status ) { } @Override public boolean isCounterStatusInitialized() { return true; } @Override public TupleEntryIterator openTapForRead( Tap tap ) throws IOException { return null; } @Override public TupleEntryCollector openTapForWrite( Tap tap ) throws IOException { return null; } @Override public TupleEntryCollector openTrapForWrite( Tap trap ) throws IOException { return null; } @Override public TupleEntryCollector openSystemIntermediateForWrite() throws IOException { return null; } @Override public Object getConfig() { return null; } @Override public Object getConfigCopy() { return null; } @Override public Object copyConfig( Object config ) { return null; } @Override public Map<String, String> diffConfigIntoMap( Object defaultConfig, Object updatedConfig ) { return null; } @Override public Object mergeMapIntoConfig( Object defaultConfig, Map<String, String> map ) { return null; } } @Test public void testLimit() { int limit = 20; int tasks = 20; int values = 10; for( int currentLimit = 0; currentLimit < limit; currentLimit++ ) { for( int currentTask = 1; currentTask < tasks; currentTask++ ) { for( int currentValue = 1; currentValue < values; currentValue++ ) { performLimitTest( currentLimit, currentTask, currentValue ); } } } } private void performLimitTest( int limit, int tasks, int values ) { Filter filter = new Limit( limit ); int count = 0; for( int i = 0; i < tasks; i++ ) { FlowProcess process = new TestFlowProcess( tasks, i ); filter.prepare( process, operationCall ); operationCall.setArguments( getEntry( new Tuple( 1 ) ) ); for( int j = 0; j < values; j++ ) { if( !filter.isRemove( process, operationCall ) ) count++; } } String message = String.format( "limit:%d tasks:%d values:%d", limit, tasks, values ); assertEquals( message, Math.min( limit, values * tasks ), count ); } }