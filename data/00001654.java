public class HadoopGroupByGate extends HadoopGroupGate { public HadoopGroupByGate ( FlowProcess flowProcess , GroupBy groupBy , IORole role ) { super ( flowProcess , groupBy , role ) ; } @ Override protected HadoopGroupByClosure createClosure ( ) { return new HadoopGroupByClosure ( flowProcess , keyFields , valuesFields ) ; } @ Override protected void wrapGroupingAndCollect ( Duct previous , int ordinal , Tuple valuesTuple , Tuple groupKey ) throws java . io . IOException { collector . collect ( groupKey , valuesTuple ) ; } @ Override protected Tuple unwrapGrouping ( Tuple key ) { return sortFields == null ? key : ( ( TuplePair ) key ) . getLhs ( ) ; } protected OutputCollector createOutputCollector ( ) { return ( ( HadoopFlowProcess ) flowProcess ) . getOutputCollector ( ) ; } }