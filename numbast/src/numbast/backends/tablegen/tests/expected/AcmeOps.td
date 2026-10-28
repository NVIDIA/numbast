#ifndef ACME_OPS_TD
#define ACME_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "acme_dialect/Dialect/Acme/IR/AcmeDialect.td"
include "acme_dialect/Dialect/Acme/IR/AcmeTypes.td"

class Acme_Op<string mnemonic, list<Trait> traits = []>
    : Op<Acme_Dialect, mnemonic, traits>;

def Acme_StreamCreateOp
    : Acme_Op<"stream.create", []> {

  let arguments = (ins I32:$Flags);
  let results = (outs Acme_StreamType:$phStream, Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeStreamCreate"); }
  }];
}

def Acme_StreamWaitEventOp
    : Acme_Op<"stream.wait.event", []> {

  let arguments = (ins Acme_StreamType:$hStream, Acme_EventType:$hEvent, I32:$Flags);
  let results = (outs Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeStreamWaitEvent"); }
  }];
}

def Acme_DeviceGetCountOp
    : Acme_Op<"device.get.count", []> {

  let arguments = (ins);
  let results = (outs I32:$count, Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeDeviceGetCount"); }
  }];
}

def Acme_VersionOp
    : Acme_Op<"version", []> {

  let arguments = (ins);
  let results = (outs I32:$ret);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeVersion"); }
  }];
}

def Acme_StreamSetLimitOp
    : Acme_Op<"stream.set.limit", []> {

  let arguments = (ins Acme_StreamType:$hStream, I64:$value_);
  let results = (outs Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeStreamSetLimit"); }
  }];
}

def Acme_StreamOpenHandleOp
    : Acme_Op<"stream.open.handle", []> {

  let arguments = (ins Acme_HandleType:$handle);
  let results = (outs Acme_StreamType:$phStream, Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeStreamOpenHandle"); }
  }];
}

def Acme_ShutdownOp
    : Acme_Op<"shutdown", []> {

  let arguments = (ins);
  let results = (outs);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeShutdown"); }
  }];
}

def Acme_GetResourceOp
    : Acme_Op<"get.resource", []> {

  let arguments = (ins Acme_StreamType:$hStream, I32:$type_);
  let results = (outs Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeGetResource"); }
  }];
}

def Acme_StreamConfigureOp
    : Acme_Op<"stream.configure", []> {

  let arguments = (ins Acme_StreamType:$hStream, Acme_DescType:$desc);
  let results = (outs Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeStreamConfigure"); }
  }];
}

def Acme_StreamOperateOp
    : Acme_Op<"stream.operate", []> {

  let arguments = (ins Acme_StreamType:$hStream, Acme_OperandType:$operand);
  let results = (outs Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeStreamOperate"); }
  }];
}

def Acme_DeviceGetResourceOp
    : Acme_Op<"device.get.resource", []> {

  let arguments = (ins Acme_DevResourceType:$resource);
  let results = (outs Acme_ResultType:$result);
  let extraClassDeclaration = [{
    std::string getAPI() { return std::string("_acmeDeviceGetResource"); }
  }];
}

#endif // ACME_OPS_TD
