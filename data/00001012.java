public class BalanceGroupSplitSpliceExpression extends RuleExpression { public static final FlowElementExpression GROUP = new FlowElementExpression ( Group . class ) ; public static final FlowElementExpression SPLICE = new FlowElementExpression ( Splice . class ) ; public BalanceGroupSplitSpliceExpression ( ) { super ( new NoSpliceTapExpressionGraph ( ) , new ExpressionGraph ( ) . arcs ( GROUP , SPLICE ) . arcs ( GROUP , SPLICE ) , new ExpressionGraph ( ) . arcs ( new FlowElementExpression ( ElementCapture . Primary , Pipe . class , TypeExpression . Topo . SplitOnly ) ) ) ; } }