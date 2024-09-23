public class LocalStepStreamGraph extends NodeStreamGraph { private LocalFlowStep step ; public LocalStepStreamGraph ( FlowProcess < Properties > flowProcess , LocalFlowStep step , FlowNode node ) { super ( flowProcess , node ) ; this . step = step ; buildGraph ( ) ; setTraps ( ) ; setScopes ( ) ; printGraph ( node . getID ( ) , "local" , 0 ) ; bind ( ) ; printBoundGraph ( node . getID ( ) , "local" , 0 ) ; } protected void buildGraph ( ) { for ( Object rhsElement : node . getSourceTaps ( ) ) { Duct rhsDuct = new SourceStage ( tapFlowProcess ( ( Tap ) rhsElement ) , ( Tap ) rhsElement ) ; addHead ( rhsDuct ) ; handleDuct ( ( FlowElement ) rhsElement , rhsDuct ) ; } } @ Override protected Duct createFork ( Duct [ ] allNext ) { return new ParallelFork ( allNext ) ; } protected Gate createCoGroupGate ( CoGroup element , IORole role ) { return new MemoryCoGroupGate ( flowProcess , element ) ; } protected Gate createGroupByGate ( GroupBy element , IORole source ) { return new LocalGroupByGate ( flowProcess , element ) ; } @ Override protected Duct createMergeStage ( Merge merge , IORole both ) { return new SyncMergeStage ( flowProcess , merge ) ; } @ Override protected SinkStage createSinkStage ( Tap element ) { return new SinkStage ( tapFlowProcess ( element ) , element ) ; } private LocalFlowProcess tapFlowProcess ( Tap tap ) { Properties defaultProperties = ( ( LocalFlowProcess ) flowProcess ) . getConfig ( ) ; Properties tapProperties = step . getPropertiesMap ( ) . get ( tap ) ; tapProperties = PropertyUtil . createProperties ( tapProperties , defaultProperties ) ; return new LocalFlowProcess ( ( LocalFlowProcess ) flowProcess , tapProperties ) ; } }