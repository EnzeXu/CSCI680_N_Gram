public class EveryAssemblyFactory extends AssemblyFactory { public Pipe createAssembly ( Pipe pipe , Fields argFields , Fields declFields , String fieldValue , Fields selectFields ) { pipe = new GroupBy ( pipe , Fields . ALL ) ; return new Every ( pipe , argFields , new TestAggregator ( declFields , new Tuple ( fieldValue ) ) , selectFields ) ; } }