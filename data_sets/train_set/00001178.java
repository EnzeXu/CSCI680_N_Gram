public class ConsecutiveGroupOrMergeExpressionGraph extends ExpressionGraph { public ConsecutiveGroupOrMergeExpressionGraph() { super( SearchOrder.Topological ); this.arc( new GroupOrMergeElementExpression( ElementCapture.Primary, TypeExpression.Topo.LinearOut ), ScopeExpression.ALL, new GroupOrMergeElementExpression() ); } }