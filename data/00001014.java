public class BalanceGroupSplitNotTapExpression extends RuleExpression { public static final FlowElementExpression SHARED_GROUP = new FlowElementExpression ( Group . class ) ; public BalanceGroupSplitNotTapExpression ( ) { super ( new SyncPipeExpressionGraph ( ) , new ExpressionGraph ( ) . arcs ( SHARED_GROUP , or ( new FlowElementExpression ( HashJoin . class ) , new FlowElementExpression ( Group . class ) ) ) . arcs ( SHARED_GROUP , or ( new FlowElementExpression ( HashJoin . class ) , new FlowElementExpression ( Group . class ) ) ) , new ExpressionGraph ( ) . arcs ( new FlowElementExpression ( ElementCapture . Primary , Pipe . class , TypeExpression . Topo . SplitOnly ) ) ) ; } }