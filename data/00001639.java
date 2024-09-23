public class FlowReducer extends MapReduceBase implements Reducer { private static final Logger LOG = LoggerFactory . getLogger ( FlowReducer . class ) ; private FlowNode flowNode ; private HadoopReduceStreamGraph streamGraph ; private HadoopFlowProcess currentProcess ; private TimedIterator < Tuple > [ ] timedIterators ; private boolean calledPrepare = false ; private HadoopGroupGate group ; private long processBeginTime ; public FlowReducer ( ) { } @ Override public void configure ( JobConf jobConf ) { try { super . configure ( jobConf ) ; HadoopUtil . initLog4j ( jobConf ) ; LOG . info ( "cascading version : { } " , jobConf . get ( "cascading . version" , "" ) ) ; LOG . info ( "child jvm opts : { } " , jobConf . get ( "mapred . child . java . opts" , "" ) ) ; currentProcess = new HadoopFlowProcess ( new FlowSession ( ) , jobConf , false ) ; timedIterators = TimedIterator . iterators ( new TimedIterator < Tuple > ( currentProcess , SliceCounters . Read_Duration , SliceCounters . Tuples_Read ) ) ; String reduceNodeState = jobConf . getRaw ( "cascading . flow . step . node . reduce" ) ; if ( reduceNodeState == null ) reduceNodeState = readStateFromDistCache ( jobConf , jobConf . get ( FlowStep . CASCADING_FLOW_STEP_ID ) , "reduce" ) ; flowNode = deserializeBase64 ( reduceNodeState , jobConf , BaseFlowNode . class ) ; LOG . info ( "flow node id : { } , ordinal : { } " , flowNode . getID ( ) , flowNode . getOrdinal ( ) ) ; streamGraph = new HadoopReduceStreamGraph ( currentProcess , flowNode , Util . getFirst ( flowNode . getSourceElements ( ) ) ) ; group = ( HadoopGroupGate ) streamGraph . getHeads ( ) . iterator ( ) . next ( ) ; for ( Duct head : streamGraph . getHeads ( ) ) LOG . info ( "sourcing from : " + ( ( ElementDuct ) head ) . getFlowElement ( ) ) ; for ( Duct tail : streamGraph . getTails ( ) ) LOG . info ( "sinking to : " + ( ( ElementDuct ) tail ) . getFlowElement ( ) ) ; for ( Tap trap : flowNode . getTraps ( ) ) LOG . info ( "trapping to : " + trap ) ; logMemory ( LOG , "flow node id : " + flowNode . getID ( ) + " , mem on start" ) ; } catch ( Throwable throwable ) { reportIfLocal ( throwable ) ; if ( throwable instanceof CascadingException ) throw ( CascadingException ) throwable ; throw new FlowException ( "internal error during reducer configuration" , throwable ) ; } } public void reduce ( Object key , Iterator values , OutputCollector output , Reporter reporter ) throws IOException { currentProcess . setReporter ( reporter ) ; currentProcess . setOutputCollector ( output ) ; timedIterators [ 0 ] . reset ( values ) ; if ( !calledPrepare ) { streamGraph . prepare ( ) ; calledPrepare = true ; processBeginTime = System . currentTimeMillis ( ) ; currentProcess . increment ( SliceCounters . Process_Begin_Time , processBeginTime ) ; currentProcess . increment ( StepCounters . Process_Begin_Time , processBeginTime ) ; group . start ( group ) ; } try { group . accept ( ( Tuple ) key , timedIterators ) ; } catch ( StopDataNotificationException exception ) { LogUtil . logWarnOnce ( LOG , "received unsupported stop data notification , ignoring : { } " , exception . getMessage ( ) ) ; } catch ( OutOfMemoryError error ) { throw error ; } catch ( Throwable throwable ) { reportIfLocal ( throwable ) ; if ( throwable instanceof CascadingException ) throw ( CascadingException ) throwable ; throw new FlowException ( "internal error during reducer execution" , throwable ) ; } } @ Override public void close ( ) throws IOException { try { if ( calledPrepare ) { group . complete ( group ) ; streamGraph . cleanup ( ) ; } super . close ( ) ; } finally { if ( currentProcess != null ) { long processEndTime = System . currentTimeMillis ( ) ; currentProcess . increment ( SliceCounters . Process_End_Time , processEndTime ) ; currentProcess . increment ( SliceCounters . Process_Duration , processEndTime - processBeginTime ) ; currentProcess . increment ( StepCounters . Process_End_Time , processEndTime ) ; currentProcess . increment ( StepCounters . Process_Duration , processEndTime - processBeginTime ) ; } String message = "flow node id : " + flowNode . getID ( ) ; logMemory ( LOG , message + " , mem on close" ) ; logCounters ( LOG , message + " , counter : " , currentProcess ) ; } } private void reportIfLocal ( Throwable throwable ) { if ( HadoopUtil . isLocal ( currentProcess . getJobConf ( ) ) ) HadoopFlowStepJob . reportLocalError ( throwable ) ; } }