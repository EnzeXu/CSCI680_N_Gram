public class TapOrBoundaryElementExpression extends OrElementExpression { public TapOrBoundaryElementExpression() { super( new FlowElementExpression( Tap.class ), new FlowElementExpression( Boundary.class ) ); } public TapOrBoundaryElementExpression( TypeExpression.Topo topo ) { super( new FlowElementExpression( Tap.class, topo ), new FlowElementExpression( Boundary.class, topo ) ); } public TapOrBoundaryElementExpression( ElementCapture capture ) { super( capture, new FlowElementExpression( Tap.class ), new FlowElementExpression( Boundary.class ) ); } public TapOrBoundaryElementExpression( ElementCapture capture, TypeExpression.Topo topo ) { super( capture, new FlowElementExpression( Tap.class, topo ), new FlowElementExpression( Boundary.class, topo ) ); } }