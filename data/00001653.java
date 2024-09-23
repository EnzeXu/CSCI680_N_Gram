public class HadoopReduceStreamGraph extends NodeStreamGraph { public HadoopReduceStreamGraph ( HadoopFlowProcess flowProcess , FlowNode node , FlowElement sourceElement ) { super ( flowProcess , node , sourceElement ) ; buildGraph ( ) ; setTraps ( ) ; setScopes ( ) ; printGraph ( node . getID ( ) , "reduce" , flowProcess . getCurrentSliceNum ( ) ) ; bind ( ) ; printBoundGraph ( node . getID ( ) , "reduce" , flowProcess . getCurrentSliceNum ( ) ) ; } protected void buildGraph ( ) { Group group = ( Group ) Util . getFirst ( node . getSourceElements ( ) ) ; Duct rhsDuct ; if ( group . isGroupBy ( ) ) rhsDuct = new HadoopGroupByGate ( flowProcess , ( GroupBy ) group , IORole . source ) ; else rhsDuct = new HadoopCoGroupGate ( flowProcess , ( CoGroup ) group , IORole . source ) ; addHead ( rhsDuct ) ; handleDuct ( group , rhsDuct ) ; } @ Override protected SinkStage createSinkStage ( Tap element ) { return new HadoopSinkStage ( flowProcess , element ) ; } protected Gate createCoGroupGate ( CoGroup element , IORole role ) { throw new IllegalStateException ( "should not happen" ) ; } @ Override protected Gate createGroupByGate ( GroupBy element , IORole role ) { throw new IllegalStateException ( "should not happen" ) ; } @ Override protected Gate createHashJoinGate ( HashJoin join ) { throw new IllegalStateException ( "should not happen" ) ; } }