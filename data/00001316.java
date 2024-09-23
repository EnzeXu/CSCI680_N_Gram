public class FlowProcessWrapper < Config > extends FlowProcess < Config > { final FlowProcess < Config > delegate ; public static FlowProcess undelegate ( FlowProcess flowProcess ) { if ( flowProcess instanceof FlowProcessWrapper ) return ( ( FlowProcessWrapper ) flowProcess ) . getDelegate ( ) ; return flowProcess ; } public FlowProcessWrapper ( FlowProcess delegate ) { this . delegate = delegate ; } public FlowProcess getDelegate ( ) { return delegate ; } @ Override public FlowProcessContext getFlowProcessContext ( ) { return delegate . getFlowProcessContext ( ) ; } @ Override public FlowProcess copyWith ( Config object ) { return delegate . copyWith ( object ) ; } @ Override public String getID ( ) { return delegate . getID ( ) ; } @ Override public FlowSession getCurrentSession ( ) { return delegate . getCurrentSession ( ) ; } @ Override public void setCurrentSession ( FlowSession currentSession ) { delegate . setCurrentSession ( currentSession ) ; } @ Override public int getNumProcessSlices ( ) { return delegate . getNumProcessSlices ( ) ; } @ Override public int getCurrentSliceNum ( ) { return delegate . getCurrentSliceNum ( ) ; } @ Override public Object getProperty ( String key ) { return delegate . getProperty ( key ) ; } @ Override public Collection < String > getPropertyKeys ( ) { return delegate . getPropertyKeys ( ) ; } @ Override public Object newInstance ( String className ) { return delegate . newInstance ( className ) ; } @ Override public void keepAlive ( ) { delegate . keepAlive ( ) ; } @ Override public void increment ( Enum counter , long amount ) { delegate . increment ( counter , amount ) ; } @ Override public void increment ( String group , String counter , long amount ) { delegate . increment ( group , counter , amount ) ; } @ Override public long getCounterValue ( Enum counter ) { return delegate . getCounterValue ( counter ) ; } @ Override public long getCounterValue ( String group , String counter ) { return delegate . getCounterValue ( group , counter ) ; } @ Override public void setStatus ( String status ) { delegate . setStatus ( status ) ; } @ Override public boolean isCounterStatusInitialized ( ) { return delegate . isCounterStatusInitialized ( ) ; } @ Override public TupleEntryIterator openTapForRead ( Tap tap ) throws IOException { return delegate . openTapForRead ( tap ) ; } @ Override public TupleEntryCollector openTapForWrite ( Tap tap ) throws IOException { return delegate . openTapForWrite ( tap ) ; } @ Override public TupleEntryCollector openTrapForWrite ( Tap trap ) throws IOException { return delegate . openTrapForWrite ( trap ) ; } @ Override public TupleEntryCollector openSystemIntermediateForWrite ( ) throws IOException { return delegate . openSystemIntermediateForWrite ( ) ; } @ Override public Config getConfig ( ) { return delegate . getConfig ( ) ; } @ Override public Config getConfigCopy ( ) { return delegate . getConfigCopy ( ) ; } @ Override public < C > C copyConfig ( C jobConf ) { return delegate . copyConfig ( jobConf ) ; } @ Override public < C > Map < String , String > diffConfigIntoMap ( C defaultConfig , C updatedConfig ) { return delegate . diffConfigIntoMap ( defaultConfig , updatedConfig ) ; } @ Override public Config mergeMapIntoConfig ( Config defaultConfig , Map < String , String > map ) { return delegate . mergeMapIntoConfig ( defaultConfig , map ) ; } @ Override public TupleEntryCollector getTrapCollectorFor ( Tap trap ) { return delegate . getTrapCollectorFor ( trap ) ; } @ Override public synchronized void closeTrapCollectors ( ) { delegate . closeTrapCollectors ( ) ; } }