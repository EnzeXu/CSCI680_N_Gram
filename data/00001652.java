public class HadoopMapStreamGraph extends NodeStreamGraph { private final Tap source ; private SourceStage streamedHead ; public HadoopMapStreamGraph ( HadoopFlowProcess flowProcess , FlowNode node , Tap source ) { super ( flowProcess , node , source ) ; this . source = source ; buildGraph ( ) ; setTraps ( ) ; setScopes ( ) ; printGraph ( node . getID ( ) , "map" , flowProcess . getCurrentSliceNum ( ) ) ; bind ( ) ; printBoundGraph ( node . getID ( ) , "map" , flowProcess . getCurrentSliceNum ( ) ) ; } public SourceStage getStreamedHead ( ) { return streamedHead ; } protected void buildGraph ( ) { streamedHead = handleHead ( this . source , flowProcess ) ; Set < Tap > tributaries = ElementGraphs . findSources ( elementGraph , Tap . class ) ; tributaries . remove ( this . source ) ; for ( Object source : tributaries ) { final HadoopFlowProcess hadoopProcess = ( HadoopFlowProcess ) flowProcess ; JobConf conf = hadoopProcess . getJobConf ( ) ; String property = conf . getRaw ( "cascading . node . accumulated . source . conf . " + Tap . id ( ( Tap ) source ) ) ; if ( property == null ) throw new IllegalStateException ( "accumulated source conf property missing for : " + ( ( Tap ) source ) . getIdentifier ( ) ) ; conf = getSourceConf ( hadoopProcess , conf , property ) ; flowProcess = new HadoopFlowProcess ( hadoopProcess , conf ) { @ Override public Reporter getReporter ( ) { return hadoopProcess . getReporter ( ) ; } } ; handleHead ( ( Tap ) source , flowProcess ) ; } } private JobConf getSourceConf ( HadoopFlowProcess flowProcess , JobConf conf , String property ) { Map < String , String > priorConf ; try { priorConf = ( Map < String , String > ) HadoopUtil . deserializeBase64 ( property , conf , HashMap . class , true ) ; } catch ( IOException exception ) { throw new FlowException ( "unable to deserialize properties" , exception ) ; } return flowProcess . mergeMapIntoConfig ( conf , priorConf ) ; } private SourceStage handleHead ( Tap source , FlowProcess flowProcess ) { SourceStage sourceDuct = new SourceStage ( flowProcess , source ) ; addHead ( sourceDuct ) ; handleDuct ( source , sourceDuct ) ; return sourceDuct ; } @ Override protected SinkStage createSinkStage ( Tap element ) { return new HadoopSinkStage ( flowProcess , element ) ; } @ Override protected Gate createCoGroupGate ( CoGroup element , IORole role ) { return new HadoopCoGroupGate ( flowProcess , element , IORole . sink ) ; } @ Override protected Gate createGroupByGate ( GroupBy element , IORole role ) { return new HadoopGroupByGate ( flowProcess , element , role ) ; } @ Override protected GroupingSpliceGate createNonBlockingJoinGate ( HashJoin join ) { return new HadoopMemoryJoinGate ( flowProcess , join ) ; } }