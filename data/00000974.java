public class NoGroupMergeBoundaryTapExpressionGraph extends ExpressionGraph { public NoGroupMergeBoundaryTapExpressionGraph ( ) { super ( not ( or ( ElementCapture . Primary , new FlowElementExpression ( Extent . class ) , new FlowElementExpression ( Group . class ) , and ( new FlowElementExpression ( Merge . class ) , not ( new AnnotationExpression ( RoleMode . Logical ) ) ) , new FlowElementExpression ( Boundary . class ) , new FlowElementExpression ( Tap . class ) ) ) ) ; } }