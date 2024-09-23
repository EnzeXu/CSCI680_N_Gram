public class LocalHadoopFlowProcess extends FlowProcessWrapper < JobConf > { FlowProcess < ? extends Properties > local ; JobConf conf ; public LocalHadoopFlowProcess ( FlowProcess < ? extends Properties > delegate ) { super ( delegate ) ; local = delegate ; } @ Override public JobConf getConfig ( ) { if ( conf == null ) conf = HadoopUtil . createJobConf ( local . getConfig ( ) ) ; return conf ; } @ Override public JobConf getConfigCopy ( ) { return new JobConf ( getConfig ( ) ) ; } @ Override public Object getProperty ( String key ) { return getConfig ( ) . get ( key ) ; } @ Override public Collection < String > getPropertyKeys ( ) { Set < String > keys = new HashSet < String > ( ) ; for ( Map . Entry < String , String > entry : getConfig ( ) ) keys . add ( entry . getKey ( ) ) ; return Collections . unmodifiableSet ( keys ) ; } }