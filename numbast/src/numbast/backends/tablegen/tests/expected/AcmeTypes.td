#ifndef ACME_TYPES_TD
#define ACME_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "acme_dialect/Dialect/Acme/IR/AcmeDialect.td"

class Acme_Type<string name, string typeMnemonic>
    : TypeDef<Acme_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Acme_ResultType : Acme_Type<"Result", "result"> {
  let summary = "AcmeResult status code";
  let description = [{
    AcmeResult status code
  }];
}

def Acme_PtrType : Acme_Type<"Ptr", "ptr"> {
  let summary = "opaque caller-owned pointer/handle (FFI)";
  let description = [{
    opaque caller-owned pointer/handle (FFI)
  }];
}

def Acme_EventType : Acme_Type<"Event", "event"> {
  let summary = "opaque acme event handle";
  let description = [{
    opaque acme event handle
  }];
}

def Acme_StreamType : Acme_Type<"Stream", "stream"> {
  let summary = "opaque acme stream handle";
  let description = [{
    opaque acme stream handle
  }];
}

def Acme_DescType : Acme_Type<"Desc", "desc"> {
  let summary = "AcmeDesc_st (struct)";
  let description = [{
    AcmeDesc_st (struct)
  }];
}

def Acme_DevResourceType : Acme_Type<"DevResource", "devresource"> {
  let summary = "AcmeDevResource_st (struct)";
  let description = [{
    AcmeDevResource_st (struct)
  }];
}

def Acme_DevresourceF2Type : Acme_Type<"DevresourceF2", "devresourcef2"> {
  let summary = "DevResource_F2 (struct)";
  let description = [{
    DevResource_F2 (struct)
  }];
}

def Acme_ExtentType : Acme_Type<"Extent", "extent"> {
  let summary = "AcmeExtent_st (struct)";
  let description = [{
    AcmeExtent_st (struct)
  }];
}

def Acme_HandleType : Acme_Type<"Handle", "handle"> {
  let summary = "AcmeHandle_st (struct)";
  let description = [{
    AcmeHandle_st (struct)
  }];
}

def Acme_LocationType : Acme_Type<"Location", "location"> {
  let summary = "AcmeLocation_st (struct)";
  let description = [{
    AcmeLocation_st (struct)
  }];
}

def Acme_OffsetType : Acme_Type<"Offset", "offset"> {
  let summary = "AcmeOffset_st (struct)";
  let description = [{
    AcmeOffset_st (struct)
  }];
}

def Acme_OperandType : Acme_Type<"Operand", "operand"> {
  let summary = "AcmeOperand_st (struct)";
  let description = [{
    AcmeOperand_st (struct)
  }];
}

def Acme_OperandArrayType : Acme_Type<"OperandArray", "operandarray"> {
  let summary = "Operand_array (struct)";
  let description = [{
    Operand_array (struct)
  }];
}

def Acme_OperandPtrType : Acme_Type<"OperandPtr", "operandptr"> {
  let summary = "Operand_ptr (struct)";
  let description = [{
    Operand_ptr (struct)
  }];
}

#endif // ACME_TYPES_TD
