public class Hadoop3TezFlowProcess extends FlowProcess < TezConfiguration > implements MapRed { final TezConfiguration configuration ; private ProcessorContext context ; private Writer writer ; public Hadoop3TezFlowProcess ( ) { this . configuration = new TezConfiguration ( ) ; } public Hadoop3TezFlowProcess ( TezConfiguration configuration ) { this . configuration = configuration ; } public Hadoop3TezFlowProcess ( FlowSession flowSession , ProcessorContext context , TezConfiguration configuration ) { super ( flowSession ) ; this . context = context ; this . configuration = configuration ; } public Hadoop3TezFlowProcess ( Hadoop3TezFlowProcess flowProcess , TezConfiguration configuration ) { super ( flowProcess ) ; this . context = flowProcess . context ; this . configuration = configuration ; } public ProcessorContext getContext ( ) { return context ; } public void setWriter ( Writer writer ) { this . writer = writer ; } @ Override public FlowProcess copyWith ( TezConfiguration configuration ) { return new Hadoop3TezFlowProcess ( this , configuration ) ; } public TezConfiguration getConfiguration ( ) { return configuration ; } @ Override public TezConfiguration getConfig ( ) { return configuration ; } @ Override public TezConfiguration getConfigCopy ( ) { return new TezConfiguration ( configuration ) ; } @ Override public int getCurrentSliceNum ( ) { return getConfiguration ( ) . getInt ( "mapred . task . partition" , 0 ) ; } @ Override public int getNumProcessSlices ( ) { return 0 ; } @ Override public Reporter getReporter ( ) { if ( context == null ) return Reporter . NULL ; return new MRTaskReporter ( context ) ; } @ Override public Object getProperty ( String key ) { return configuration . get ( key ) ; } @ Override public Collection < String > getPropertyKeys ( ) { Set < String > keys = new HashSet < String > ( ) ; for ( Map . Entry < String , String > entry : configuration ) keys . add ( entry . getKey ( ) ) ; return Collections . unmodifiableSet ( keys ) ; } @ Override public Object newInstance ( String className ) { if ( className == null || className . isEmpty ( ) ) return null ; try { Class type = ( Class ) Hadoop3TezFlowProcess . class . getClassLoader ( ) . loadClass ( className . toString ( ) ) ; return ReflectionUtils . newInstance ( type , configuration ) ; } catch ( ClassNotFoundException exception ) { throw new CascadingException ( "unable to load class : " + className . toString ( ) , exception ) ; } } @ Override public void keepAlive ( ) { } @ Override public void increment ( Enum counter , long amount ) { if ( context != null ) context . getCounters ( ) . findCounter ( counter ) . increment ( amount ) ; } @ Override public void increment ( String group , String counter , long amount ) { if ( context != null ) context . getCounters ( ) . findCounter ( group , counter ) . increment ( amount ) ; } @ Override public long getCounterValue ( Enum counter ) { if ( context == null ) return 0 ; return context . getCounters ( ) . findCounter ( counter ) . getValue ( ) ; } @ Override public long getCounterValue ( String group , String counter ) { if ( context == null ) return 0 ; return context . getCounters ( ) . findCounter ( group , counter ) . getValue ( ) ; } @ Override public void setStatus ( String status ) { } @ Override public boolean isCounterStatusInitialized ( ) { if ( context == null ) return false ; return context . getCounters ( ) != null ; } @ Override public TupleEntryIterator openTapForRead ( Tap tap ) throws IOException { return tap . openForRead ( this ) ; } @ Override public TupleEntryCollector openTapForWrite ( Tap tap ) throws IOException { return tap . openForWrite ( this , null ) ; } @ Override public TupleEntryCollector openTrapForWrite ( Tap trap ) throws IOException { TezConfiguration jobConf = new TezConfiguration ( getConfiguration ( ) ) ; int stepNum = jobConf . getInt ( "cascading . flow . step . num" , 0 ) ; int nodeNum = jobConf . getInt ( "cascading . flow . node . num" , 0 ) ; String partname = String . format ( "-%05d-%05d-" , stepNum , nodeNum ) ; jobConf . set ( "cascading . tapcollector . partname" , "%s%spart" + partname + "%05d" ) ; return trap . openForWrite ( new Hadoop3TezFlowProcess ( this , jobConf ) , null ) ; } @ Override public TupleEntryCollector openSystemIntermediateForWrite ( ) throws IOException { return null ; } @ Override public < C > C copyConfig ( C config ) { return HadoopUtil . copyJobConf ( config ) ; } @ Override public < C > Map < String , String > diffConfigIntoMap ( C defaultConfig , C updatedConfig ) { return HadoopUtil . getConfig ( ( Configuration ) defaultConfig , ( Configuration ) updatedConfig ) ; } @ Override public TezConfiguration mergeMapIntoConfig ( TezConfiguration defaultConfig , Map < String , String > map ) { return HadoopUtil . mergeConf ( new TezConfiguration ( defaultConfig ) , map , true ) ; } }