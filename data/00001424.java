public class EveryAfterBufferAssert extends RuleAssert { public EveryAfterBufferAssert ( ) { super ( PreBalanceAssembly , new EveryAfterBufferExpression ( ) , "only one Every with a Buffer may follow a GroupBy or CoGroup pipe , no other Every instances are allowed immediately before or after , found : { Secondary } before : { Primary } " ) ; } }