public class BalanceGroupSplitExpression extends RuleExpression { public static final FlowElementExpression SHARED_GROUP = new FlowElementExpression ( Group . class , Topo . Split ) ; public BalanceGroupSplitExpression ( ) { super ( new SyncPipeExpressionGraph ( ) , new ExpressionGraph ( ) . arc ( SHARED_GROUP , ScopeExpression . ANY , or ( new FlowElementExpression ( HashJoin . class ) , new FlowElementExpression ( Group . class ) , new FlowElementExpression ( Tap . class ) , new FlowElementExpression ( Checkpoint . class ) ) ) . arc ( SHARED_GROUP , ScopeExpression . ANY , or ( new FlowElementExpression ( HashJoin . class ) , new FlowElementExpression ( Group . class ) , new FlowElementExpression ( Tap . class ) , new FlowElementExpression ( Checkpoint . class ) ) ) , new ExpressionGraph ( ) . arcs ( new FlowElementExpression ( ElementCapture . Primary , Pipe . class , Topo . SplitOnly ) ) ) ; } }