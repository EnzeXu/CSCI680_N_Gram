public class EachAssemblyFactory extends AssemblyFactory { public Pipe createAssembly( Pipe pipe, Fields argFields, Fields declFields, String fieldValue, Fields selectFields ) { return new Each( pipe, argFields, new TestFunction( declFields, new Tuple( fieldValue ) ), selectFields ); } }