public class NestedSetFunction<Node, Result> extends NestedBaseFunction<Node, Result> { public NestedSetFunction( NestedCoercibleType<Node, Result> nestedCoercibleType, Fields fieldDeclaration ) { super( nestedCoercibleType, fieldDeclaration ); } public NestedSetFunction( NestedCoercibleType<Node, Result> nestedCoercibleType, Fields fieldDeclaration, String rootPointer ) { super( nestedCoercibleType, fieldDeclaration, rootPointer ); } public NestedSetFunction( NestedCoercibleType nestedCoercibleType, Fields fieldDeclaration, Map<Fields, String> pointerMap ) { super( nestedCoercibleType, fieldDeclaration, pointerMap ); } @Override protected Node getNode( TupleEntry arguments ) { Node node = (Node) arguments.getObject( 0, getCoercibleType() ); return deepCopy( node ); } }