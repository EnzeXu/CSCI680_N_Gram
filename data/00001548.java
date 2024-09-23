public class OperationExpression extends TypeExpression<Operation> { public OperationExpression( ElementCapture capture, boolean exact, Class<? extends Operation> type ) { super( capture, exact, type ); } public OperationExpression( ElementCapture capture, Class<? extends Operation> type ) { super( capture, type ); } public OperationExpression( boolean exact, Class<? extends Operation> type ) { super( exact, type ); } public OperationExpression( Class<? extends Operation> type ) { super( type ); } @Override protected Class<? extends Operation> getType( FlowElement flowElement ) { if( !( flowElement instanceof Operator ) ) return null; return ( (Operator) flowElement ).getOperation().getClass(); } }